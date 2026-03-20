import re
from dataclasses import dataclass, field
from typing import List

from .ast import OverlayDecl, ConstraintDecl, BudgetDecl


@dataclass
class EPLProgram:
    overlays: List[OverlayDecl] = field(default_factory=list)
    constraints: List[ConstraintDecl] = field(default_factory=list)
    budgets: List[BudgetDecl] = field(default_factory=list)


OVERLAY_RE = re.compile(r"^overlay\s+(\w+)\s+(eventual|causal|ordered)\s*;\s*$", re.I)
CONSTRAINT_RE = re.compile(r"^constraint\s+(hard|soft)\s+(\w+)\s*:\s*(.+?)(?:\s+weight=(\d+(?:\.\d+)?))?\s*;\s*$", re.I)
BUDGET_RE = re.compile(r"^budget\s+(\w+)\s*:\s*total=(\d+(?:\.\d+)?)\s+region=(\d+)\s*;\s*$", re.I)


def parse_epl(text: str) -> EPLProgram:

    program = EPLProgram()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        overlay_match = OVERLAY_RE.match(line)
        if overlay_match:
            name, consistency = overlay_match.groups()
            program.overlays.append(OverlayDecl(name=name, consistency=consistency.lower()))
            continue

        constraint_match = CONSTRAINT_RE.match(line)
        if constraint_match:
            kind, name, expr, weight = constraint_match.groups()
            weight_val = float(weight) if weight is not None else None
            program.constraints.append(
                ConstraintDecl(name=name, kind=kind.lower(), expr=expr, weight=weight_val)
            )
            continue

        budget_match = BUDGET_RE.match(line)
        if budget_match:
            name, total, region = budget_match.groups()
            program.budgets.append(
                BudgetDecl(name=name, total=float(total), region=int(region))
            )
            continue

        raise ValueError(f"Unrecognized EPL line: {raw_line}")

    return program
