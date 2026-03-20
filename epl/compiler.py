from dataclasses import dataclass, field
from typing import List

from .parser import EPLProgram


@dataclass
class GovernanceSpec:
    hard_constraints: List[str] = field(default_factory=list)
    soft_constraints: List[str] = field(default_factory=list)
    overlays: List[str] = field(default_factory=list)
    budgets: List[str] = field(default_factory=list)


def compile_program(program: EPLProgram) -> GovernanceSpec:

    spec = GovernanceSpec()

    for constraint in program.constraints:
        if constraint.kind == "hard":
            spec.hard_constraints.append(constraint.expr)
        else:
            expr = constraint.expr
            if constraint.weight is not None:
                expr = f"{expr} (weight={constraint.weight})"
            spec.soft_constraints.append(expr)

    for overlay in program.overlays:
        spec.overlays.append(f"{overlay.name}:{overlay.consistency}")

    for budget in program.budgets:
        spec.budgets.append(f"{budget.name}:total={budget.total},region={budget.region}")

    return spec
