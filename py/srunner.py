import subprocess
from pathlib import Path
import struct
import os
import uuid
import dataclasses
import contextlib
from primitives import *
import logging
import socket
import math

logger = logging.getLogger(__name__)

@dataclasses.dataclass(frozen=True)
class RunInfo:
    solver: Path
    args: list[str]
    cnf_path: Path
    run_id: uuid.UUID 


class Connection:
    def __init__(self, addr: Path):
        self.addr = addr
        self.num_reads = 0

        if self.addr.exists():
            self.addr.unlink()
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(str(self.addr))
        self.server.listen(1)

        self.conn: socket.socket | None = None

    def start(self, routes: dict[str, callable], run_info, data: any):
        assert self.conn is None
        logger.debug("Starting connection to %s", str(self.addr))

        self.conn, _ = self.server.accept()

        while True:
            try:
                request = self.read_str()
            except EOFError:
                return

            if request not in routes:
                raise KeyError(f"Got unknown request through pipe: {request}")

            fn = routes[request]
            fn(self, run_info, data)
            self.flush()

    def read(self, n: int) -> bytes:
        self.num_reads += 1
        # print(f"called read {self.num_reads}")
        left = n
        data = b""
        while left > 0:
            chunk = self.conn.recv(left)
            if not chunk:
                raise EOFError("Connection closed")
            data += chunk
            left -= len(chunk)
        return data

    def write(self, data: bytes):
        self.conn.send(data)

    def flush(self):
        pass

    def read_u8(self) -> int:
        return struct.unpack("B", self.read(1))[0]

    def read_u16(self) -> int:
        return struct.unpack("H", self.read(2))[0]

    def read_u32(self) -> int:
        return struct.unpack("I", self.read(4))[0]

    def read_u64(self) -> int:
        return struct.unpack("Q", self.read(8))[0]
    
    def read_i8(self) -> int:
        return struct.unpack("b", self.read(1))[0]

    def read_i16(self) -> int:
        return struct.unpack("h", self.read(2))[0]

    def read_i32(self) -> int:
        return struct.unpack("i", self.read(4))[0]

    def read_i64(self) -> int:
        return struct.unpack("q", self.read(8))[0]

    def read_f32(self) -> float:
        result = struct.unpack("f", self.read(4))[0]
        assert not math.isnan(result)
        return result

    def read_f64(self) -> float:
        result = struct.unpack("d", self.read(8))[0]
        assert not math.isnan(result)
        return result

    def read_clause(self) -> Clause:
        id_ = self.read_u64()
        flags = self.read_u64()
        lbd = self.read_i32()
        size = self.read_u32()
        lits = [self.read_i32() for _ in range(size)]
        activity = self.read_f32()
        conflicts_on_creation = self.read_f32()
        times_reason = self.read_u32()
        return Clause(
            id_,
            lits,
            redundant=bool(flags & 1),
            keep=bool(flags & 2),
            used_recently=bool(flags & 4),
            lbd=lbd,
            activity=activity,
            conflicts_on_creation=conflicts_on_creation,
            times_reason=times_reason,
        )

    def read_vec_i32(self) -> list[int]:
        n = self.read_u32()
        return [self.read_i32() for _ in range(n)]

    def read_str_raw(self, n: int) -> str:
        return (self.read(n)).decode("utf-8")

    def read_str(self) -> str:
        n = self.read_u32()
        return self.read_str_raw(n)

    def write_u8(self, n: int):
        self.write(struct.pack("B", n))

    def write_u16(self, n: int):
        self.write(struct.pack("H", n))

    def write_u32(self, n: int):
        self.write(struct.pack("I", n))

    def write_u64(self, n: int):
        self.write(struct.pack("Q", n))

    def write_i8(self, n: int):
        self.write(struct.pack("b", n))

    def write_i16(self, n: int):
        self.write(struct.pack("h", n))

    def write_i32(self, n: int):
        self.write(struct.pack("I", n))

    def write_i64(self, n: int):
        self.write(struct.pack("q", n))

    def write_f32(self, n: float):
        self.write(struct.pack("f", n))

    def write_f64(self, n: float):
        self.write(struct.pack("d", n))

    def write_str_raw(self, s: str):
        self.write(s)

    def write_str(self, s: str):
        self.write_u32(len(s))
        self.write(s.encode("utf-8"))

    def write_ok(self):
        self.write_str("ok")


class Router:
    def __init__(self):
        self.routes = {}

    def route(self, name: str):
        def decorator(fn):
            self.routes[name] = fn
            return fn
        return decorator


def run_instance(
    solver: Path,
    args: list[str],
    cnf_path: Path, 
    routes: dict[str, callable],
    *,
    silent: bool = True,
    timeout_seconds: int | None = None,
    data: any = None,
    valgrind: bool = False,
):
    assert solver.exists(), f"{solver} does not exist"
    assert cnf_path.exists(), f"{cnf_path} does not exist"

    logger.info("Running %s on %s", solver, cnf_path)

    socket_path = Path(f"/tmp/{uuid.uuid4()}.sock")

    run_info = RunInfo(
        solver=solver,
        args=args,
        cnf_path=cnf_path,
        run_id=uuid.uuid4(),
    )

    connection = Connection(socket_path)

    args = [
        str(solver.absolute()),
        *args,
        "--socket", str(socket_path.absolute()),
        "-t", str(timeout_seconds) if timeout_seconds else str(10 ** 9),
        str(cnf_path.absolute()),
    ]

    if valgrind:
        args = ["valgrind"] + args

    logger.info("Running %s", " ".join(args))

    process = subprocess.Popen(
        args,
        stdout=subprocess.DEVNULL if silent else None,
        stderr=subprocess.DEVNULL if silent else None,
    )

    connection.start(routes, run_info, data)

    process.wait()
    if process.returncode is None:
        print(f"Solver {solver} timed out and has not terminated. Killing it.")
        process.terminate()
        process.wait()

    return run_info, True if process.returncode == 10 else False if process.returncode == 20 else None
