import random
import numpy as np
from collections import deque, namedtuple


class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        fields = ["state", "action", "reward", "next_state", "done"]
        self.experience = namedtuple("Experience", field_names=fields)
        self.memory = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        dones = np.vstack([e.done for e in experiences if e is not None])

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
