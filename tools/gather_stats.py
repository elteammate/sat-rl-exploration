import asyncio
import runner
from pathlib import Path


cnfs = map(Path, [
    "./instances/pigeon-7.cnf",
    "./instances/pigeon-7.cnf",
    "./instances/pigeon-7.cnf",
    "./instances/pigeon-7.cnf",
])

router = runner.Router()

@router.route("stats")
async def stats(conn: runner.Connection):
    result = {}
    while (name := await conn.read_str()) != "end":
        if name == "time":
            result[name] = await conn.read_f64()
        else:
            result[name] = await conn.read_u64()
    print(result)
    await conn.write_ok()


async def main():
    await asyncio.gather(*[
        runner.run_instance(
            Path("cadical/build/cadical"),
            [],
            cnf_path,
            silent=True,
            timeout_seconds=5,
            routes=router.routes,
        )
        for cnf_path in cnfs
    ])


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    loop.run_until_complete(main())
