import cnfgen
import random
import uuid
import asyncio
import json
from runner import *
from pathlib import Path


NUM_WORKERS = 20

result_file = open("archives/runs.json", "a")


def uniform_random_3cnf(seed) -> cnfgen.CNF:
    random.seed(seed)
    nvars = random.randrange(200, 400)
    cnf = cnfgen.RandomKCNF(3, nvars, int(nvars * 4.3), seed=seed)
    return cnf


async def main():
    queue = asyncio.Queue(NUM_WORKERS)

    async def worker_func():
        cnf_path: Path = await queue.get()

        print(f"Running {cnf_path}")

        router = Router()

        stats = None

        @router.route("stats")
        async def stats(conn, run_info, data):
            nonlocal stats
            result = {}
            while (name := await conn.read_str()) != "end":
                if name == "time":
                    result[name] = await conn.read_f64()
                else:
                    result[name] = await conn.read_u64()
            stats = result
            await conn.write_ok()

        _, result1 = await run_instance(
            Path("cadical/build/cadical"),
            ["--reduce-mode", "0"],
            cnf_path,
            router.routes,
            silent=True,
            timeout_seconds=10
        )

        if result1 is None:
            return

        stats1 = stats

        _, result2 = await run_instance(
            Path("cadical/build/cadical"),
            ["--reduce-mode", "1"],
            cnf_path,
            router.routes,
            silent=True,
            timeout_seconds=10,
        )

        stats2 = stats

        if result2 is None:
            return

        if stats1["reductions"] + 5 > stats2["reductions"]:
            print(f"Skipping {cnf_path} due to too few reductions: {stats1['reductions']} vs {stats2['reductions']}")
            return

        saved_path = Path(f"./instances/") / cnf_path.name
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        saved_path.write_bytes(cnf_path.read_bytes())

        result = {
            "path": str(saved_path.absolute()),
            "result": result1,
            "stats": stats1,
            "stats_no_reductions": stats2,
        }

        result_file.write(json.dumps(result) + "\n")

        queue.task_done()

    async def worker():
        while True:
            await worker_func()

    workers = [asyncio.create_task(worker()) for _ in range(NUM_WORKERS)]

    while True:
        uid = uuid.uuid4()
        problem = uniform_random_3cnf(uid.hex)

        path = Path(f"/tmp/instances/unif-{uid}.cnf")
        path.parent.mkdir(parents=True, exist_ok=True)
        problem.to_file(str(path.absolute()))

        await queue.put(path)


if __name__ == "__main__":
    import asyncio
    loop = asyncio.new_event_loop()
    loop.run_until_complete(main())
