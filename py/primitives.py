import dataclasses


@dataclasses.dataclass(frozen=True)
class Clause:
    id_: int
    lits: list[int]
