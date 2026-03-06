import random
from collections import deque


class ReplayBuffer:

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, next_state):
        self.buffer.append((state.detach(), action.detach(), next_state.detach()))

    def sample(self, batch_size):

        batch = random.sample(self.buffer, batch_size)

        states, actions, next_states = zip(*batch)

        return states, actions, next_states

    def __len__(self):
        return len(self.buffer)