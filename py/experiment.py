import asyncio
from runner import *
from pathlib import Path


cnfs = map(Path, [
    "./instances/pigeon-11.cnf",
    # "./instances/pigeon-7.cnf",
    # "./instances/pigeon-7.cnf",
    # "./instances/pigeon-7.cnf",
])

router = Router()

@router.route("stats")
async def stats(conn: Connection, info: RunInfo, data):
    result = {}
    while (name := await conn.read_str()) != "end":
        if name == "time":
            result[name] = await conn.read_f64()
        else:
            result[name] = await conn.read_u64()
    print(result)
    await conn.write_ok()


@router.route("reduce")
async def reduce(conn: Connection, info: RunInfo, data):
    num_clauses = await conn.read_u64()
    clauses = [await conn.read_clause() for _ in range(num_clauses)]
    num_reducible = await conn.read_u64()
    num_target = await conn.read_u64()
    reducible_ids = [await conn.read_u64() for _ in range(num_reducible)]

    print(clauses, reducible_ids)



async def main():
    await asyncio.gather(*[
        run_instance(
            Path("cadical/build/cadical"),
            ["--reduce-mode", "0"],
            cnf_path,
            router.routes,
            silent=True,
            timeout_seconds=1,
        )
        for cnf_path in cnfs
    ])


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    loop.run_until_complete(main())
