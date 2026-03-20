from dataclasses import dataclass
from typing import Optional


@dataclass
class OverlayDecl:
    name: str
    consistency: str


@dataclass
class ConstraintDecl:
    name: str
    kind: str  # "hard" or "soft"
    expr: str
    weight: Optional[float] = None


@dataclass
class BudgetDecl:
    name: str
    total: float
    region: int
