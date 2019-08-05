from game.engine import Agent, Card
import numpy as np
from keras.models import Sequential, Model
from keras.layers import *
from keras import backend as KTF
import random
from collections import deque, defaultdict
import pickle as pkl
import os


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
    #S.append(list_to_mat(state.self_out))
    #S.append(list_to_mat(state.up_out))
    #S.append(list_to_mat(state.down_out))
    #S.append(list_to_mat(state.other_hand))
    S.append(list_to_mat(state.last_move))
    S.append(list_to_mat([4]*13+[1,1]))
    S = np.array(S).transpose([1,2,0])
    return S   # 15*4*8

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

class ReplayBufferHER():
    def __init__(self, maxlen=None, K=4):
        self.memory = deque(maxlen=maxlen)
        self.maxlen = maxlen
        self.round_exp = []
        self.K = K
    
    def __len__(self):
        return len(self.memory)

    def remember(self, *data):
        state, move, reward, next_state, next_move, goal, done = data
        self.round_exp.append([state, move, reward, next_state, next_move, goal, done])

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        batch = list(zip(*batch))
        data = []
        for i in range(len(batch)):
            data.append(np.asarray(batch[i]))
        return data

    def update_memory(self):
        self.memory.extend(self.round_exp)
        for t in range(len(self.round_exp)):
            for k in range(self.K):
                future = np.random.randint(t, len(self.round_exp))
                goal = self.round_exp[future][3][:, :, 0]  # next_state of future
                state = self.round_exp[t][0]
                move = self.round_exp[t][1]
                next_state = self.round_exp[t][3]
                next_move = self.round_exp[t][4]
                done = np.array_equal(next_state[:, :, 0], goal)
                reward = 100 if done else -100
                self.memory.append([state, move, reward, next_state, next_move, goal, done])

        self.round_exp = []

class DQNModel():
    def __init__(self, epsilon=1.0, min_epsilon = 0.02, epsilon_decay=0.99, gamma=0.95, buf_size=10000, step_per_update = 10, weight_blend = 0.8):
        self.buf = ReplayBuffer(buf_size)
        self.use_HER = False
        self.max_epsilon = self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.qnet, self.tnet = self.__build_model()
        self.step_per_update = step_per_update
        self.weight_blend = weight_blend
        self.__J = 0

    def __build_model(self):
        def bottleneck(inp, channels, use_shortcut = False):
            in_shape = KTF.int_shape(inp)
            conv1 = Conv2D(channels, (2,2), activation='relu', padding='same')(inp)
            conv2 = Conv2D(channels, (2,2), activation='relu', padding='same')(conv1)
            if use_shortcut:
                shortcut = inp
                if in_shape[3]!=channels:
                    shortcut = Conv2D(channels, (1,1))(inp)
                plus = Add()([conv2, shortcut])
                return Activation('relu')(plus)
            return conv2

        def build_net(use_HER):
            state_in = Input(shape=(15, 4, 4))
            action_in = Input(shape=(15, 4))
            action = Reshape((15, 4, 1))(action_in)
            if use_HER:
                goal_in = Input(shape=(15, 4))
                goal = Reshape((15, 4, 1))(goal_in)
                In = Concatenate()([state_in, action, goal])
                inputs = [state_in, action_in, goal_in]
            else:
                In = Concatenate()([state_in, action])
                inputs = [state_in, action_in]

            conv1 = Conv2D(256, (1,1), strides=(1,4))(In)
            conv2 = Conv2D(256, (1,2), strides=(1,4))(In)
            conv3 = Conv2D(256, (1,3), strides=(1,4))(In)
            conv4 = Conv2D(256, (1,4), strides=(1,4))(In)
            conca = Concatenate(axis=-2)([conv1, conv2, conv3, conv4])
            pool = MaxPool2D((1,4))(conca)
            out = Flatten()(pool)
            out = Dropout(0.5)(out)
            out = Dense(256)(out)
            out = Dense(1)(out)
            m = Model(inputs=inputs, outputs=[out])
            return m
        qnet = build_net(self.use_HER)
        qnet.compile(loss='mse', optimizer='adam')
        tnet = build_net(self.use_HER)
        tnet.set_weights(qnet.get_weights())
        tnet.trainable = False
        qnet.summary()
        return qnet, tnet

    def choose_action(self, state, valid_actions, goal=None, ignore_eps=False):
        if not ignore_eps and np.random.rand()<self.epsilon:
            i = np.random.choice(len(valid_actions))
            return valid_actions[i]
        q = self.qnet.predict(make_input(state, valid_actions, goal)).reshape(-1,)
        i = np.argmax(q)
        return valid_actions[i]

    def learn(self, batch_size):
        state, move, reward, next_state, next_move, done = self.buf.sample(batch_size)
        q_hat = self.tnet.predict([next_state, next_move]).reshape(-1,)
        target = reward + (1.0-done)*self.gamma*q_hat

        self.qnet.train_on_batch([state, move], target)
        self.__J+=1
        if self.__J>=self.step_per_update:
            self.__J = 0
            #self.tnet.set_weights( (1-self.weight_blend)*self.tnet.get_weights() + self.weight_blend*self.qnet.get_weights() )
            self.tnet.set_weights(self.qnet.get_weights())
            #if self.epsilon>self.min_epsilon:
            #    self.epsilon*=self.epsilon_decay

    def update_epsilon(self, episode, max_episode):
        if self.epsilon>self.min_epsilon:
            self.epsilon = self.max_epsilon - 3*(self.max_epsilon-self.min_epsilon)/max_episode * episode

    def save(self):
        self.qnet.save('qnet.h5')
        pkl.dump({'epsilon':self.epsilon},open('config.pkl', 'wb'))

class DQNAgent(Agent):
    def __init__(self, player_id, rl_model):
        super().__init__(player_id)
        self.model = rl_model

    def choose(self, state):
        move_list = self.move_list
        res = self.model.choose_action(state_to_tensor(state), move_list)
        if os.path.exists( os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test')):
            # player i [手牌] // [出牌]
            hand_card = []
            for i, n in enumerate(Card.all_card_name):
                hand_card.extend([n]*self.get_hand_card()[i])
            print("DQN Player {}".format(self.player_id), ' ', hand_card, ' // ', Card.visual_card(res))

        return res, None


