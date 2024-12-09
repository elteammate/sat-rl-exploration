import dataclasses


@dataclasses.dataclass(frozen=True)
class Clause:
    id_: int
    lits: list[int]
    lbd: int
    redundant: bool
    keep: bool
    used_recently: bool
    activity: float = 0.0
    conflicts_on_creation: float = 0.0
    times_reason: int = 0

    def with_lits(self, lits):
        return dataclasses.replace(self, lits=lits)
