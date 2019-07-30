import random
from collections import deque
import numpy as np


class ReplayBuffer():
    def __init__(self, maxlen=None):
        self.memory = deque(maxlen=maxlen)
        self.maxlen = maxlen
        self.round_exp = []
    
    def __len__(self):
        return len(self.memory)

    def remember(self, state, action, reward, next_state, done, goal):
        self.round_exp.append([state, action, reward, next_state, done, goal])

    def sample(self, batch_size):
        """
        returns: state, action, reward, next_state, done, goal
        """
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones, goals = zip(*batch)

        states  = np.asarray(states)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        next_states = np.asarray(next_states)
        dones = np.asarray(dones, dtype=np.uint8)
        goals = np.asarray(goals)

        return states, actions, rewards, next_states, dones, goals

    def update_memory(self):
        self.memory.extend(self.round_exp)
        self.round_exp = []


class ReplayBufferHER(ReplayBuffer):
    def __init__(self, maxlen=None, K = 4):
        super().__init__(maxlen=maxlen)
        self.K = K

    def update_memory(self):
        self.memory.extend(self.round_exp)
        for t in range(len(self.round_exp)):
            for k in range(self.K):
                future = np.random.randint(t, len(self.round_exp))
                goal = self.round_exp[future][3]  # next_state of future
                state = self.round_exp[t][0]
                action = self.round_exp[t][1]
                next_state = self.round_exp[t][3]
                done = np.array_equal(next_state, goal)
                reward = 0 if done else -1
                self.memory.append([state, action, reward, next_state, done, goal])

        self.round_exp = []

