import numpy as np


class GridWorld:

    def __init__(self, size=5):

        self.size = size
        self.goal_pos = [size - 1, size - 1]

        self.reset()


    # -----------------------------------------

    def reset(self):

        self.agent_pos = [0, 0]

        return self.get_state()


    # -----------------------------------------

    def get_state(self):

        grid = np.zeros((self.size, self.size))

        ax, ay = self.agent_pos
        gx, gy = self.goal_pos

        grid[ax][ay] = 1
        grid[gx][gy] = 2

        return grid.flatten()


    # -----------------------------------------

    def distance_to_goal(self):

        ax, ay = self.agent_pos
        gx, gy = self.goal_pos

        return abs(ax - gx) + abs(ay - gy)


    # -----------------------------------------

    def step(self, action):

        old_distance = self.distance_to_goal()

        # actions
        # 0 = up
        # 1 = down
        # 2 = left
        # 3 = right

        if action == 0:
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)

        elif action == 1:
            self.agent_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)

        elif action == 2:
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)

        elif action == 3:
            self.agent_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)

        new_distance = self.distance_to_goal()

        # -----------------------------------------
        # Reward shaping
        # -----------------------------------------

        reward = -0.1 + (old_distance - new_distance)

        done = False

        # goal reached
        if self.agent_pos == self.goal_pos:

            reward = 10
            done = True

        return self.get_state(), reward, done


    # -----------------------------------------

    def render(self):

        grid = [["." for _ in range(self.size)] for _ in range(self.size)]

        ax, ay = self.agent_pos
        gx, gy = self.goal_pos

        grid[ax][ay] = "A"
        grid[gx][gy] = "G"

        for row in grid:
            print(" ".join(row))

        print()