import numpy as np


class RegionalBudgetAllocator:

    def __init__(
        self,
        grid_size,
        total_budget=1.0,
        smoothing=0.2,
        error_weight=0.7,
        visit_weight=0.3
    ):

        self.grid_size = grid_size
        self.total_budget = total_budget
        self.smoothing = smoothing

        self.error_weight = error_weight
        self.visit_weight = visit_weight

        self.budget_map = np.full((grid_size, grid_size), total_budget / (grid_size * grid_size))
        self.visit_map = np.zeros((grid_size, grid_size))

    # --------------------------------------------------

    def record_visit(self, position):

        x, y = position
        self.visit_map[x][y] += 1

    # --------------------------------------------------

    def update(self, error_map):

        error_map = np.array(error_map, dtype=np.float32)
        visit_map = np.array(self.visit_map, dtype=np.float32)

        error_norm = error_map / (error_map.max() + 1e-6)
        visit_norm = visit_map / (visit_map.max() + 1e-6)

        utility = self.error_weight * error_norm + self.visit_weight * visit_norm
        utility = utility / (utility.sum() + 1e-6)

        new_budget = utility * self.total_budget

        self.budget_map = (
            (1.0 - self.smoothing) * self.budget_map + self.smoothing * new_budget
        )

    # --------------------------------------------------

    def set_total_budget(self, total_budget):

        self.total_budget = float(total_budget)
        self.budget_map = np.full(
            (self.grid_size, self.grid_size),
            self.total_budget / (self.grid_size * self.grid_size)
        )

    # --------------------------------------------------

    def get_budget_scale(self, position):

        x, y = position
        max_budget = self.budget_map.max()

        if max_budget <= 0:
            return 0.5

        return float(self.budget_map[x][y] / max_budget)
