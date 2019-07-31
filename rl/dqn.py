from game.engine import Agent
from .replay_buffer import ReplayBuffer
import numpy as np
from keras.models import Sequential, Model
from keras.layers import *
from keras import backend as K

def build_model():
    m = Sequential()
    return m

class DQNAgent(Agent):
    def __init__(self, player_id, rl_model):
        super().__init__(player_id)
        self.model = rl_model
        self.buffer = ReplayBuffer(1000)

    def choose(self, state):
        return []

    def learn(self, batch_size = 128, **kwargs):
        pass

    def store_transition(self, *data):
        self.buffer.remember(*data)

