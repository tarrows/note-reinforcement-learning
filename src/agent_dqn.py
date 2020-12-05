import random
import numpy as np
from q_network import q_network
from replay_buffer import ReplayBuffer

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
UPDATE_EVERY = 4
GAMMA = 0.99


class AgentDQN:
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.qnet_local = q_network(state_size, action_size, seed)
        self.qnet_target = q_network(state_size, action_size, seed)

        self.qnet_local = self.qnet_local.compile(optimizer='adam')
        self.qnet_target = self.qnet_target.compile(optimizer='adam')

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        if random.random() > eps:
            return np.argmax([])
        else:
            return random.choice(np.arange(self.action_size))

    def _learn(self):
        pass

    def _soft_update(self):
        pass
