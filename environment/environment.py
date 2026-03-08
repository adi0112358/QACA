import numpy as np
import random


class GridWorld:

    def __init__(self, size=5):

        self.size = size
        self.goal_pos = [size - 1, size - 1]

        self.reset()

    # ---------------------------------

    def reset(self):

        # learning agent
        self.agentA_pos = [0, 0]

        # second agent (environment agent)
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

        return grid.flatten()

    # ---------------------------------

    def distance_to_goal(self):

        ax, ay = self.agentA_pos
        gx, gy = self.goal_pos

        return abs(ax - gx) + abs(ay - gy)

    # ---------------------------------

    def move_agent(self, pos, action):

        x, y = pos

        if action == 0:  # up
            x = max(0, x - 1)

        elif action == 1:  # down
            x = min(self.size - 1, x + 1)

        elif action == 2:  # left
            y = max(0, y - 1)

        elif action == 3:  # right
            y = min(self.size - 1, y + 1)

        return [x, y]

    # ---------------------------------

    def move_agentB(self):
        """
        Simple reactive policy for second agent.
        Moves randomly.
        """

        action = random.randint(0, 3)
        self.agentB_pos = self.move_agent(self.agentB_pos, action)

    # ---------------------------------

    def move_goal(self):
        """
        Introduce environment perturbation.
        Goal moves occasionally.
        """

        if self.step_count % 5 == 0:

            gx = random.randint(0, self.size - 1)
            gy = random.randint(0, self.size - 1)

            self.goal_pos = [gx, gy]

    # ---------------------------------

    def step(self, action):

        self.step_count += 1

        old_distance = self.distance_to_goal()

        # move learning agent
        self.agentA_pos = self.move_agent(self.agentA_pos, action)

        # move environment agent
        self.move_agentB()

        # move goal occasionally
        self.move_goal()

        new_distance = self.distance_to_goal()

        # reward shaping
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

        for row in grid:
            print(" ".join(row))

        print()