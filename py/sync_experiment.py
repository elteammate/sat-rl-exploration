from srunner import *
import pickle
from pathlib import Path

cnfs = map(Path, [
    "/home/elt/projects/sat-rl-exploration/instances/unif-4b56bc27-c539-4dd9-aab4-5b551be4be0c.cnf",
    # "./instances/pigeon-7.cnf",
    # "./instances/pigeon-7.cnf",
    # "./instances/pigeon-7.cnf",
])

router = Router()

@router.route("stats")
def stats(conn: Connection, info: RunInfo, data):
    result = {}
    while (name := conn.read_str()) != "end":
        if name == "time":
            result[name] = conn.read_f64()
        else:
            result[name] = conn.read_u64()
    print(result)
    conn.write_ok()


@dataclasses.dataclass
class ReductionProblem:
    num_vars: int
    levels: list[int]
    vals: list[int]
    clauses: list[list[Clause]]
    reducible_ids: list[int]
    conflicts: int


first_time = True


@router.route("reduce")
def reduce(conn: Connection, info: RunInfo, data):
    global first_time

    num_vars = conn.read_u64()
    levels = [-1] * num_vars
    vals = [-1] * num_vars
    for i in range(num_vars):
        vals[i] = conn.read_i8()
        levels[i] = conn.read_i32()
    num_clauses = conn.read_u64()
    clauses = [conn.read_clause() for _ in range(num_clauses)]
    num_reducible = conn.read_u64()
    num_target = conn.read_u64()
    reducible_ids = [conn.read_u64() for _ in range(num_reducible)]
    conflicts = conn.read_u64()

    problem = ReductionProblem(
        num_vars=num_vars,
        levels=levels,
        vals=vals,
        clauses=clauses,
        reducible_ids=reducible_ids,
        conflicts=conflicts,
    )

    if first_time:
        with open("archives/example-problem.pkl", "wb") as f:
            pickle.dump(problem, f)
        first_time = False

    # print(num_vars, levels, vals, clauses, num_reducible, num_target, reducible_ids)
    print(num_vars)
    conn.write_ok()
    conn.write_u32(1)
    conn.write_u64(reducible_ids[0])


def main():
    for cnf_path in cnfs:
        run_instance(
            Path("cadical/build/cadical"),
            ["--reduce-mode", "2"],
            cnf_path,
            router.routes,
            silent=False,
            timeout_seconds=100,
        )


if __name__ == "__main__":
    main()
