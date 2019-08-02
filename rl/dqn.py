from game.engine import Agent
import numpy as np
from keras.models import Sequential, Model
from keras.layers import *
from keras import backend as K
import random
from collections import deque, defaultdict

class ReplayBuffer():
    def __init__(self, maxlen=None):
        self.memory = deque(maxlen=maxlen)
        self.maxlen = maxlen
        self.round_exp = []
    
    def __len__(self):
        return len(self.memory)

    def remember(self, *data):
        self.round_exp.append(*data)

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        batch = list(zip(*batch))
        data = []
        for i in range(len(data)):
            data.append(np.asarray(batch[i]))
        return data

    def update_memory(self):
        self.memory.extend(self.round_exp)
        self.round_exp = []

def list_to_mat(lst):
    m = np.zeros((15,4))
    for i in range(len(lst)):
        if lst[i]>0:
            m[i, :lst[i]] = 1
    return m

def state_to_tensor(state):
    S = []
    S.append(list_to_mat(state.out))
    S.append(list_to_mat(state.hand))
    S.append(list_to_mat(state.self_out))
    S.append(list_to_mat(state.up_out))
    S.append(list_to_mat(state.down_out))
    S.append(list_to_mat(state.other_hand))
    S.append(list_to_mat(state.last_move))
    # S: 7*15*4
    return S


def make_tnet_input(state_batch, moves_batch):
    #state -> matrix
    S_batch = []
    for state in state_batch:
        S_batch.append(state_to_tensor(state)) # S_batch: None*7*15*4

    index = []
    data = []
    for i in range(len(S_batch)):
        for j in range(len(moves_batch[i])):
            index.append(i)
            a = np.array(S_batch[i]+[ list_to_mat(moves_batch[i][j]) ])# a: 8*15*4
            data.append( np.transpose(a, [1,2,0]) ) #trans a: 15*4*8
    return np.array(index), np.array(data) #data: None*15*4*8

def make_qnet_input(state_batch, move_batch):
    S_batch = []
    for i in range(len(state_batch)):
        tmp = state_to_tensor(state_batch[i]) # tmp: 7*15*4
        S_batch.append( tmp+[ list_to_mat(move_batch[i]) ] )# append: 8*15*4
    a = np.array(S_batch) #a: None*8*15*4
    return np.transpose(a, [0,2,3,1])   #return: None*15*4*8


class DQNModel():
    def __init__(self, epsilon=1.0, min_epsilon = 0.02, epsilon_decay=0.99, gamma=0.95, buf_size=10000, step_per_update = 10, weight_blend = 0.8):
        self.buf = ReplayBuffer(buf_size)
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.qnet, self.tnet = self.__build_model()
        self.step_per_update = step_per_update
        self.weight_blend = weight_blend
        self.__J = 0

    def __build_model(self):
        def build_net():
            m = Sequential()
            m.add(Conv2D(64, kernel_size = (3,4), activation='relu', padding='SAME', input_shape=(15,4,8)))
            m.add(Conv2D(64, kernel_size = (7,4), activation='relu', padding='SAME') )
            m.add(Flatten())
            m.add(Dense(256, activation='relu'))
            m.add(Dense(1))
            return m
        qnet = build_net()
        tnet = build_net()
        tnet.set_weights(qnet.get_weights())
        tnet.trainable = False
        return qnet, tnet

    def choose_action(self, state, valid_actions):
        _, inputs = make_tnet_input([state], [valid_actions])
        q = self.qnet.predict(inputs).reshape(-1,)
        i = np.argmax(q)
        return valid_actions[i]

    def learn(self, batch_size):
        state, _, action, reward, next_state, next_moves = self.buf.sample(batch_size)
        index, tinputs = make_tnet_input(next_state, next_moves)
        q_hat = self.tnet.predict(tinputs)
        d = defaultdict(list)
        for i in range(index):
            d[i].append(q_hat)
        target = np.zeros(len(d))
        for k,v in d.items():
            target[k] = max(v)
        target = reward + self.gamma*target
        self.qnet.train_on_batch( make_qnet_input(state, action), target )
        self.__J+=1
        if self.__J>=self.step_per_update:
            self.__J = 0
            self.tnet.set_weights( (1-self.weight_blend)*self.tnet.get_weights() + self.weight_blend*self.qnet.get_weights() )
            if self.epsilon>self.min_epsilon:
                self.epsilon*=self.epsilon_decay


    def store_transition(self, state, valid_actions, action, reward, next_state, next_moves):
        self.buf.remember([state, valid_actions, action, reward, next_state, next_moves])


class DQNAgent(Agent):
    def __init__(self, player_id, rl_model):
        super().__init__(player_id)
        self.model = rl_model

    def choose(self, state):
        move_list = self.move_list
        res = self.model.choose_action(state, move_list)
        return res, None


