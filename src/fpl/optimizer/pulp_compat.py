import warnings
from collections.abc import Iterable

import pulp  # type: ignore[import-untyped]


def binary_variables(
    problem: pulp.LpProblem, prefix: str, indices: Iterable[int]
) -> dict[int, pulp.LpVariable]:
    return {i: problem.add_variable(f"{prefix}_{i}", cat=pulp.LpBinary) for i in indices}


def cbc_solver() -> pulp.LpSolver:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="PULP_CBC_CMD is deprecated")
        return pulp.PULP_CBC_CMD(msg=False)
