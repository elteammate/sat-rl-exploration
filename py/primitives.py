import dataclasses


@dataclasses.dataclass(frozen=True)
class Clause:
    id_: int
    lits: list[int]
    lbd: int
    redundant: bool
    keep: bool
    used_recently: bool

    def with_lits(self, lits):
        return dataclasses.replace(self, lits=lits)
