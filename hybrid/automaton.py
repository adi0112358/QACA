from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


GuardFn = Callable[[Dict[str, float]], bool]
ResetFn = Callable[[Dict[str, float]], Dict[str, float]]


@dataclass
class Mode:
    name: str
    invariants: List[str] = field(default_factory=list)


@dataclass
class Transition:
    source: str
    target: str
    guard: GuardFn
    reset: Optional[ResetFn] = None
    label: str = ""


class HybridAutomaton:

    def __init__(self, modes: List[Mode], transitions: List[Transition], initial_mode: str):
        self.modes = {m.name: m for m in modes}
        self.transitions = transitions
        self.mode = initial_mode

    # --------------------------------------------------

    def step(self, context: Dict[str, float]) -> Tuple[str, Optional[str]]:
        """
        Evaluate transitions from current mode. If any guard is satisfied,
        transition to the first matching target.
        Returns (mode, transition_label)
        """

        for t in self.transitions:
            if t.source != self.mode:
                continue
            if t.guard(context):
                if t.reset is not None:
                    context.update(t.reset(context))
                self.mode = t.target
                return self.mode, t.label or f"{t.source}->{t.target}"

        return self.mode, None
