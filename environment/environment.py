import numpy as np
import random


class GridWorld:

    def __init__(
        self,
        size=5,
        obstacle_prob=0.1,
        agent_b_policy="random",
        pursuit_prob=0.7,
        perturb_prob=0.02,
        goal_move_period=5
    ):

        self.size = size
        self.obstacle_prob = obstacle_prob
        self.agent_b_policy = agent_b_policy
        self.pursuit_prob = pursuit_prob
        self.perturb_prob = perturb_prob
        self.goal_move_period = goal_move_period

        self._base_agent_b_policy = agent_b_policy
        self._base_pursuit_prob = pursuit_prob
        self._base_perturb_prob = perturb_prob
        self._base_goal_move_period = goal_move_period

        self.goal_pos = [size - 1, size - 1]

        # obstacle grid
        self.obstacles = np.zeros((size, size))

        self.reset()

    # ---------------------------------

    def generate_obstacles(self):

        self.obstacles = np.zeros((self.size, self.size))

        for i in range(self.size):
            for j in range(self.size):

                if random.random() < self.obstacle_prob:
                    self.obstacles[i][j] = 1

        # ensure start and goal are free
        self.obstacles[0][0] = 0
        gx, gy = self.goal_pos
        self.obstacles[gx][gy] = 0

    # ---------------------------------

    def reset(self):

        # generate new obstacle layout
        self.generate_obstacles()

        # learning agent
        self.agentA_pos = [0, 0]

        # second agent
        self.agentB_pos = [self.size - 1, 0]

        self.step_count = 0

        return self.get_state()

    # ---------------------------------

    def get_state(self):

        grid = np.zeros((self.size, self.size))

        ax, ay = self.agentA_pos
        bx, by = self.agentB_pos
        gx, gy = self.goal_pos

        grid[ax][ay] = 1     # Agent A
        grid[bx][by] = 2     # Agent B
        grid[gx][gy] = 3     # Goal

        # mark obstacles
        for i in range(self.size):
            for j in range(self.size):
                if self.obstacles[i][j] == 1:
                    grid[i][j] = -1

        return grid.flatten()

    # ---------------------------------

    def distance_to_goal(self):

        ax, ay = self.agentA_pos
        gx, gy = self.goal_pos

        return abs(ax - gx) + abs(ay - gy)

    # ---------------------------------

    def move_agent(self, pos, action):

        x, y = pos

        new_x, new_y = x, y

        if action == 0:  # up
            new_x = max(0, x - 1)

        elif action == 1:  # down
            new_x = min(self.size - 1, x + 1)

        elif action == 2:  # left
            new_y = max(0, y - 1)

        elif action == 3:  # right
            new_y = min(self.size - 1, y + 1)

        # prevent walking into obstacle
        if self.obstacles[new_x][new_y] == 1:
            return pos

        return [new_x, new_y]

    # ---------------------------------

    def move_agentB(self):

        if self.agent_b_policy == "pursue":
            if random.random() < self.pursuit_prob:
                action = self._pursuit_action(self.agentB_pos, self.agentA_pos)
            else:
                action = random.randint(0, 3)
        else:
            action = random.randint(0, 3)

        self.agentB_pos = self.move_agent(self.agentB_pos, action)

    # ---------------------------------

    def _pursuit_action(self, src, tgt):

        sx, sy = src
        tx, ty = tgt

        if abs(tx - sx) > abs(ty - sy):
            return 1 if tx > sx else 0

        if ty != sy:
            return 3 if ty > sy else 2

        return random.randint(0, 3)

    # ---------------------------------

    def move_goal(self):

        if self.step_count % max(1, self.goal_move_period) == 0:

            gx = random.randint(0, self.size - 1)
            gy = random.randint(0, self.size - 1)

            # avoid obstacle cell
            if self.obstacles[gx][gy] == 0:
                self.goal_pos = [gx, gy]

    # ---------------------------------

    def perturb_obstacles(self):

        if random.random() > self.perturb_prob:
            return

        for i in range(self.size):
            for j in range(self.size):
                if random.random() < 0.02:
                    self.obstacles[i][j] = 1 - self.obstacles[i][j]

        # keep key positions clear
        self.obstacles[0][0] = 0
        ax, ay = self.agentA_pos
        bx, by = self.agentB_pos
        gx, gy = self.goal_pos
        self.obstacles[ax][ay] = 0
        self.obstacles[bx][by] = 0
        self.obstacles[gx][gy] = 0

    # ---------------------------------

    def apply_governance(self, safe_mode=False, intensity=0.0):

        intensity = max(0.0, min(1.0, float(intensity)))

        if safe_mode:
            self.agent_b_policy = "random"
            self.pursuit_prob = max(0.1, self._base_pursuit_prob * 0.3)
            self.perturb_prob = 0.0
            self.goal_move_period = max(self._base_goal_move_period, 20)
            return

        self.agent_b_policy = self._base_agent_b_policy
        self.pursuit_prob = self._base_pursuit_prob * (1.0 - 0.3 * intensity)
        self.perturb_prob = self._base_perturb_prob * (1.0 - 0.5 * intensity)
        self.goal_move_period = max(1, int(round(self._base_goal_move_period * (1.0 + 0.5 * intensity))))

    # ---------------------------------

    def step(self, action):

        self.step_count += 1

        old_distance = self.distance_to_goal()

        # move learning agent
        self.agentA_pos = self.move_agent(self.agentA_pos, action)

        # move second agent
        self.move_agentB()

        # dynamic goal
        self.move_goal()

        # perturb environment
        self.perturb_obstacles()

        new_distance = self.distance_to_goal()

        reward = -0.1 + (old_distance - new_distance)

        done = False

        # collision penalty
        if self.agentA_pos == self.agentB_pos:
            reward -= 2

        # goal reached
        if self.agentA_pos == self.goal_pos:
            reward = 10
            done = True

        return self.get_state(), reward, done

    # ---------------------------------

    def render(self):

        grid = [["." for _ in range(self.size)] for _ in range(self.size)]

        ax, ay = self.agentA_pos
        bx, by = self.agentB_pos
        gx, gy = self.goal_pos

        grid[ax][ay] = "A"
        grid[bx][by] = "B"
        grid[gx][gy] = "G"

        for i in range(self.size):
            for j in range(self.size):

                if self.obstacles[i][j] == 1:
                    grid[i][j] = "#"

        for row in grid:
            print(" ".join(row))

        print()
