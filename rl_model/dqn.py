from game.engine import Agent, Card
import numpy as np
import random
from collections import deque, defaultdict
import pickle as pkl
import os
from .architectures import MiniResNet, SimpleConv
from keras.models import load_model

def list_to_mat(lst):
    m = np.zeros((15,4))
    for i in range(len(lst)):
        if lst[i]>0:
            m[i, :lst[i]] = 1
    return m

def state_to_tensor(state):
    S = []
    S.append(list_to_mat(state.hand))
    S.append(list_to_mat(state.out))
    S.append(list_to_mat(state.self_out))
    S.append(list_to_mat(state.up_out))
    S.append(list_to_mat(state.down_out))
    S.append(list_to_mat(state.other_hand))
    S.append(list_to_mat(state.last_move))
    S.append(list_to_mat([4]*13+[1,1]))
    S = np.array(S)
    role = np.zeros((3,15,4))
    role[state.player_role, :, :] = 1
    S = np.concatenate([S, role], axis=0).transpose([1,2,0])
    return S   # 15*4*11

def make_input(state, move_list, goal=None):
    S = np.array([state]*len(move_list))
    M = np.array([list_to_mat(m) for m in move_list])
    if goal is not None:
        G = np.array([goal]*len(move_list))
        return [S, M, G]
    return [S, M]

class ReplayBuffer():
    def __init__(self, maxlen=None):
        self.memory = deque(maxlen=maxlen)
        self.maxlen = maxlen
        self.round_exp = []
    
    def __len__(self):
        return len(self.memory)

    def remember(self, *data):
        state, action, reward, next_state, next_action, done = data
        self.round_exp.append([state, action, reward, next_state, next_action, done])

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        batch = list(zip(*batch))
        data = []
        for i in range(len(batch)):
            data.append(np.asarray(batch[i]))
        return data

    def update_memory(self):
        self.memory.extend(self.round_exp)
        self.round_exp = []

class DQNModel():
    def __init__(self, state_shape, epsilon=1.0, min_epsilon = 0.02, epsilon_decay=0.99, gamma=0.95, buf_size=10000, step_per_update = 10, weight_blend = 0.8):
        self.buf = ReplayBuffer(buf_size)
        self.use_HER = False
        self.max_epsilon = self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.step_per_update = step_per_update
        self.weight_blend = weight_blend
        self.__J = 0
        self.state_shape = state_shape
        self.qnet, self.tnet = self.__build_model()
        
    def __build_model(self):
        qnet = SimpleConv(self.state_shape, self.use_HER)
        tnet = SimpleConv(self.state_shape, self.use_HER)
        qnet.compile(loss='mse', optimizer='adam')
        qnet.summary()
        tnet.set_weights(qnet.get_weights())
        tnet.trainable = False
        return qnet, tnet

    def choose_action(self, state, valid_actions, goal=None, ignore_eps=False):  # if inputs change
        if not ignore_eps and np.random.rand()<self.epsilon:
            i = np.random.choice(len(valid_actions))
            return valid_actions[i]
        q = self.qnet.predict(make_input(state, valid_actions, goal)).reshape(-1,)  # if inputs change
        i = np.argmax(q)
        return valid_actions[i]

    def learn(self, batch_size):
        state, move, reward, next_state, next_move, done = self.buf.sample(batch_size)  # if inputs change
        q_hat = self.tnet.predict([next_state, next_move]).reshape(-1,)  # if inputs change
        target = reward + (1.0-done)*self.gamma*q_hat

        self.qnet.train_on_batch([state, move], target)  # if inputs change
        self.__J+=1
        if self.__J>=self.step_per_update:
            self.__J = 0
            self.tnet.set_weights(self.qnet.get_weights())

    def update_epsilon(self, episode, max_episode):
        if self.epsilon>self.min_epsilon:
            self.epsilon = self.max_epsilon - 3*(self.max_epsilon-self.min_epsilon)/max_episode * episode

    def save(self, filename=None):
        if filename is None:
            filename = 'qnet.h5'
        self.qnet.save(filename)

    def load(self, filename=None):
        if filename is None:
            filename = 'qnet.h5'
        self.qnet = load_model(filename)
        self.tnet.set_weights(self.qnet.get_weights())

class DQNAgent(Agent):
    def __init__(self, player_id, rl_model):
        super().__init__(player_id)
        self.model = rl_model

    def choose(self, state):
        move_list = self.move_list
        res = self.model.choose_action(state_to_tensor(state), move_list)
        if os.path.exists( os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test')):
            # player i [手牌] // [出牌]
            print("DQN Player {}".format(self.player_id), ' ', Card.visual_card(self.get_hand_card()), ' // ', Card.visual_card(res))

        return res, None


