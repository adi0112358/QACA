import numpy as np
import matplotlib.pyplot as plt


class PredictionErrorHeatmap:

    def __init__(self, grid_size):

        self.grid_size = grid_size
        self.error_map = np.zeros((grid_size, grid_size))
        self.count_map = np.zeros((grid_size, grid_size))


    def update(self, agent_position, error):

        x, y = agent_position

        self.error_map[x][y] += error
        self.count_map[x][y] += 1


    def get_average_error(self):

        avg_map = np.zeros_like(self.error_map)

        for i in range(self.grid_size):
            for j in range(self.grid_size):

                if self.count_map[i][j] > 0:
                    avg_map[i][j] = self.error_map[i][j] / self.count_map[i][j]

        return avg_map


    def plot(self):

        avg_map = self.get_average_error()

        plt.figure(figsize=(6,6))
        plt.imshow(avg_map, cmap="hot", interpolation="nearest")
        plt.colorbar(label="Prediction Error")

        plt.title("World Model Prediction Error Heatmap")
        plt.xlabel("Y")
        plt.ylabel("X")

        plt.show()