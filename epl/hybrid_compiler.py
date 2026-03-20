from dataclasses import dataclass
from typing import Dict

from hybrid import Mode, Transition, HybridAutomaton
from .parser import EPLProgram


@dataclass
class HybridCompileConfig:
    divergence_threshold: float = 1.5
    recovery_threshold: float = 0.8


def compile_to_automaton(program: EPLProgram, config: HybridCompileConfig) -> HybridAutomaton:
    """
    Minimal EPL -> Hybrid Automaton compilation.
    Creates two modes: normal and safe. Hard constraints are treated as invariants.
    Transitions are guarded by divergence and any hard barrier violation.
    """

    hard_invariants = [c.expr for c in program.constraints if c.kind == "hard"]

    normal = Mode(name="normal", invariants=hard_invariants)
    safe = Mode(name="safe", invariants=hard_invariants)

    def to_safe(context: Dict[str, float]) -> bool:
        if context.get("barrier_violation", 0.0) > 0:
            return True
        return context.get("divergence", 0.0) >= config.divergence_threshold

    def to_normal(context: Dict[str, float]) -> bool:
        return context.get("divergence", 0.0) <= config.recovery_threshold and context.get("barrier_violation", 0.0) <= 0

    transitions = [
        Transition(source="normal", target="safe", guard=to_safe, label="normal->safe"),
        Transition(source="safe", target="normal", guard=to_normal, label="safe->normal")
    ]

    return HybridAutomaton([normal, safe], transitions, initial_mode="normal")
