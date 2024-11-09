from pathlib import Path
import struct
import os
import uuid
import asyncio


class Connection:
    def __init__(self, addr: Path):
        self.addr = addr
        self.s2p: asyncio.StreamReader = None
        self.p2s: asyncio.StreamWriter = None
        self.ready = asyncio.Event()

    async def start(self):
        if self.addr.exists():
            self.addr.unlink()

        client_connected = asyncio.Event()

        async def _handle_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
            if self.s2p:
                raise ValueError("Connection already established")
            self.s2p = reader
            self.p2s = writer
            client_connected.set()

        self.server = await asyncio.start_unix_server(_handle_connection, path=str(self.addr))
        self.ready.set()

        self.serving_task = asyncio.create_task(self.server.serve_forever())
        await client_connected.wait()

    async def close(self):
        if self.p2s:
            self.p2s.close()
            await self.p2s.wait_closed()

        if self.addr.exists():
            self.addr.unlink()

        self.serving_task.cancel()

    async def read(self, n: int) -> bytes:
        return await self.s2p.readexactly(n)

    async def write(self, data: bytes):
        self.p2s.write(data)

    async def flush(self):
        await self.p2s.drain()

    async def read_u8(self) -> int:
        return struct.unpack("B", await self.read(1))[0]

    async def read_u16(self) -> int:
        return struct.unpack("H", await self.read(2))[0]

    async def read_u32(self) -> int:
        return struct.unpack("I", await self.read(4))[0]

    async def read_u64(self) -> int:
        return struct.unpack("Q", await self.read(8))[0]

    async def read_i8(self) -> int:
        return struct.unpack("b", await self.read(1))[0]

    async def read_i16(self) -> int:
        return struct.unpack("h", await self.read(2))[0]

    async def read_i32(self) -> int:
        return struct.unpack("i", await self.read(4))[0]

    async def read_i64(self) -> int:
        return struct.unpack("q", await self.read(8))[0]

    async def read_f32(self) -> float:
        return struct.unpack("f", await self.read(4))[0]

    async def read_f64(self) -> float:
        return struct.unpack("d", await self.read(8))[0]

    async def read_str_raw(self, n: int) -> str:
        return (await self.read(n)).decode("utf-8")

    async def read_str(self) -> str:
        n = await self.read_u32()
        return await self.read_str_raw(n)

    async def write_u8(self, n: int):
        await self.write(struct.pack("B", n))

    async def write_u16(self, n: int):
        await self.write(struct.pack("H", n))

    async def write_u32(self, n: int):
        await self.write(struct.pack("I", n))

    async def write_u64(self, n: int):
        await self.write(struct.pack("Q", n))

    async def write_i8(self, n: int):
        await self.write(struct.pack("b", n))

    async def write_i16(self, n: int):
        await self.write(struct.pack("h", n))

    async def write_i32(self, n: int):
        await self.write(struct.pack("I", n))

    async def write_i64(self, n: int):
        await self.write(struct.pack("q", n))

    async def write_f32(self, n: float):
        await self.write(struct.pack("f", n))

    async def write_f64(self, n: float):
        await self.write(struct.pack("d", n))

    async def write_str_raw(self, s: str):
        await self.write(s)

    async def write_str(self, s: str):
        await self.write_u32(len(s))
        await self.write(s.encode("utf-8"))

    async def write_ok(self):
        await self.write_str("ok")


class Router:
    def __init__(self):
        self.routes = {}

    def route(self, name: str):
        def decorator(fn):
            self.routes[name] = fn
            return fn
        return decorator


async def handle_pipe(
    connection: Connection,
    routes: dict[str, callable],
):
    await connection.start()

    while True:
        request = await connection.read_str()
        if request not in routes:
            raise KeyError(f"Got unknown request through pipe: {request}")

        fn = routes[request]
        if asyncio.iscoroutinefunction(fn):
            await fn(connection)
        else:
            fn(connection)
        await connection.flush()


async def run_instance(
    solver: Path,
    args: list[str],
    cnf_path: Path, *,
    silent: bool = True,
    timeout_seconds: int | None = None,
    routes: dict[str, callable],
):
    assert solver.exists(), f"{solver} does not exist"
    assert cnf_path.exists(), f"{cnf_path} does not exist"

    socket_path = Path(f"/tmp/{uuid.uuid4()}.sock")

    connection = Connection(socket_path)
    task = asyncio.create_task(handle_pipe(connection, routes))

    args = [
        str(solver.absolute()),
        *args,
        "--socket", str(socket_path.absolute()),
        "-t", str(timeout_seconds) if timeout_seconds else "0",
        str(cnf_path.absolute()),
    ]
    print(*args)

    await connection.ready.wait()
    process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.DEVNULL if silent else None,
        stderr=asyncio.subprocess.DEVNULL if silent else None,
    )
    # print(f"Solver finished with {await process.wait()}")

    try:
        await asyncio.wait_for(process.communicate(), timeout=timeout_seconds + 1)
    except asyncio.TimeoutError:
        print(f"Solver {solver} timed out and has not terminated. Killing it.")
        process.terminate()
        await process.wait()
    finally:
        task.cancel()
        await connection.close()
