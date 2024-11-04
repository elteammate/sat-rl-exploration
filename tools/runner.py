from pathlib import Path
import struct
import os
import uuid
import asyncio


class Connection:
    def __init__(self, fifo_p2s: Path, fifo_s2p: Path, delete_on_close: bool = False):
        self.fifo_p2s = fifo_p2s
        self.fifo_s2p = fifo_s2p
        self.delete_on_close = delete_on_close

    def __enter__(self):
        self.p2s = open(self.fifo_p2s, "rb")
        os.set_blocking(self.p2s.fileno(), False)
        self.s2p = open(self.fifo_s2p, "wb")
        os.set_blocking(self.s2p.fileno(), False)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.p2s.close()
        self.s2p.close()
        if self.delete_on_close:
            self.fifo_p2s.unlink()
            self.fifo_s2p.unlink()

    async def read(self, n: int) -> bytes:
        data = b""
        while len(data) < n:
            data += self.p2s.read(n - len(data))
            print(f"Read {data}")
            await asyncio.sleep(0)
        return data

    async def write(self, data: bytes):
        written = 0
        while written < len(data):
            written += self.s2p.write(data[written:])
            await asyncio.sleep(0)

    def flush(self):
        self.s2p.flush()

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
        return (await self.read(n)).decode("utf-8")

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
    with connection:
        while True:
            request_len = await connection.read_u32()
            if request_len == 0:
                raise ValueError("Got empty request through pipe")
            if request_len > 32:
                raise ValueError("Request is too long")
            request = await connection.read_str_raw(request_len)
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
    cnf_path: Path, *,
    timeout_seconds: int | None = None,
    routes: dict[str, callable],
):
    assert solver.exists(), f"{solver} does not exist"
    assert cnf_path.exists(), f"{cnf_path} does not exist"

    fifo_s2p = Path(f"/tmp/{uuid.uuid4()}.fifo")
    fifo_p2s = Path(f"/tmp/{uuid.uuid4()}.fifo")
    os.mkfifo(fifo_s2p)
    os.mkfifo(fifo_p2s)

    connection = Connection(fifo_p2s, fifo_s2p, delete_on_close=True)

    args = [
        str(solver.absolute()),
        "--pipe-in", str(fifo_p2s.absolute()),
        "--pipe-out", str(fifo_s2p.absolute()),
        "-t", str(timeout_seconds) if timeout_seconds else "0",
        str(cnf_path.absolute()),
    ]

    process = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    task = asyncio.create_task(handle_pipe(connection, routes))
    # print(f"Solver finished with {await process.wait()}")

    try:
        await asyncio.wait_for(process.communicate(), timeout=timeout_seconds + 1)
    except asyncio.TimeoutError:
        print(f"Solver {solver} timed out and has not terminated. Killing it.")
        process.terminate()
        await process.wait()
    finally:
        task.cancel()
