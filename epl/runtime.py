import ast
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

from .parser import EPLProgram


@dataclass
class EPLRuntimeConfig:
    beta: float = 5.0
    default_weight: float = 1.0


class EPLRuntime:

    def __init__(self, program: EPLProgram, config: Optional[EPLRuntimeConfig] = None):
        self.program = program
        self.config = config or EPLRuntimeConfig()

        self._compiled = []

        for constraint in program.constraints:
            compiled = _compile_constraint(constraint.expr)
            self._compiled.append(
                (
                    constraint.name,
                    constraint.kind,
                    constraint.weight if constraint.weight is not None else self.config.default_weight,
                    compiled
                )
            )

    # --------------------------------------------------

    def evaluate(self, context: Dict[str, float]) -> Tuple[float, Dict[str, float]]:

        soft_penalty = 0.0
        hard_barriers: Dict[str, float] = {}

        for name, kind, weight, compiled in self._compiled:
            result = compiled(context)

            if kind == "hard":
                hard_barriers[name] = result
            else:
                penalty = _soft_penalty(-result, self.config.beta) * weight
                soft_penalty += penalty

        return soft_penalty, hard_barriers


# ------------------------------------------------------


def _soft_penalty(x, beta):
    if x <= 0:
        return 0.0
    scaled = beta * x
    if scaled > 50:
        return x
    return math.log1p(math.exp(scaled)) / beta


def _compile_constraint(expr: str):

    tree = ast.parse(expr, mode="eval").body

    if isinstance(tree, ast.Compare) and len(tree.ops) == 1 and len(tree.comparators) == 1:
        left = tree.left
        op = tree.ops[0]
        right = tree.comparators[0]

        def _compiled(context):
            left_val = ast_to_float(left, context)
            right_val = ast_to_float(right, context)
            if isinstance(op, (ast.Lt, ast.LtE)):
                return right_val - left_val
            if isinstance(op, (ast.Gt, ast.GtE)):
                return left_val - right_val
            raise ValueError("Unsupported comparator in EPL constraint")

        return _compiled

    def _compiled_default(context):
        value = ast_to_float(tree, context)
        return -value

    return _compiled_default


def ast_to_float(node, context: Optional[Dict[str, float]] = None):

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("EPL constants must be numeric")

    if isinstance(node, ast.Name):
        if context is None or node.id not in context:
            raise ValueError(f"Unknown EPL variable: {node.id}")
        return float(context[node.id])

    if isinstance(node, ast.BinOp):
        left = ast_to_float(node.left, context)
        right = ast_to_float(node.right, context)

        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left ** right
        if isinstance(node.op, ast.Mod):
            return left % right

        raise ValueError("Unsupported binary operator in EPL expression")

    if isinstance(node, ast.UnaryOp):
        operand = ast_to_float(node.operand, context)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError("Unsupported unary operator in EPL expression")

    raise ValueError("Unsupported EPL expression syntax")
