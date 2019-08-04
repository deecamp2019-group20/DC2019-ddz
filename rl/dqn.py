from game.engine import Agent, Card
import numpy as np
from keras.models import Sequential, Model
from keras.layers import *
from keras import backend as K
import random
from collections import deque, defaultdict
import pickle as pkl
import os

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
                goal = self.round_exp[future][3].hand  # next_state of future
                state = self.round_exp[t][0]
                move = self.round_exp[t][1]
                next_state = self.round_exp[t][3]
                next_move = self.round_exp[t][4]
                done = np.array_equal(next_state.hand, goal)
                reward = 100 if done else -100
                self.memory.append([state, move, reward, next_state, next_move, goal, done])

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
    S.append(list_to_mat([4]*13+[1,1]))
    # S: 8*15*4
    return S


def merge_input(state_batch, moves_batch, goal_batch):
    #state -> matrix
    S_batch = []
    for state in state_batch:
        S_batch.append(state_to_tensor(state)) # S_batch: None*7*15*4

    data = []
    for i in range(len(S_batch)):
        for j in range(len(moves_batch[i])):
            a = np.array(S_batch[i]+[ list_to_mat(moves_batch[i][j]), list_to_mat(goal_batch[i]) ])# a: 10*15*4
            data.append( np.transpose(a, [1,2,0]) ) #trans a: 15*4*10
    return np.array(data) #data: None*15*4*10

def make_qnet_input(state_batch, move_batch, goal_batch):
    S_batch = []
    for i in range(len(state_batch)):
        tmp = state_to_tensor(state_batch[i]) # tmp: 8*15*4
        tmp+=[ list_to_mat(move_batch[i]), list_to_mat(goal_batch[i]) ] 
        S_batch.append( tmp )# append: 10*15*4
    a = np.array(S_batch) #a: None*10*15*4
    return np.transpose(a, [0,2,3,1])   #return: None*15*4*10


class DQNModel():
    def __init__(self, epsilon=1.0, min_epsilon = 0.02, epsilon_decay=0.99, gamma=0.95, buf_size=10000, step_per_update = 10, weight_blend = 0.8):
        self.buf = ReplayBufferHER(buf_size)
        self.max_epsilon = self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.qnet, self.tnet = self.__build_model()
        self.step_per_update = step_per_update
        self.weight_blend = weight_blend
        self.__J = 0

    def __build_model(self):
        def build_net():
            In = Input(shape=(15, 4, 10))
            conv1 = Conv2D(128, (1, 4), activation='relu', padding='SAME')(In)
            conv2 = Conv2D(128, (3, 4), activation='relu', padding='SAME')(In)
            conv5 = Conv2D(128, (5, 4), activation='relu', padding='SAME')(In)
            conv9 = Conv2D(128, (9, 4), activation='relu', padding='SAME')(In)
            conv15 =Conv2D(128, (15,4), activation='relu', padding='SAME')(In)
            conc = Concatenate()([conv1, conv2, conv5, conv9, conv15])  #None x 15 x 4 x all channel
            conv1x1 = Conv2D(512, (1,1))(conc)
            bnorm = BatchNormalization()(conv1x1)
            acti = Activation('relu')(bnorm)
            out = Flatten()(acti)
            out = Dense(128, activation='relu')(out)
            out = Dense(1)(out)
            m = Model(inputs=[In], outputs=[out])
            return m
        qnet = build_net()
        qnet.compile(loss='mse', optimizer='adam')
        tnet = build_net()
        tnet.set_weights(qnet.get_weights())
        tnet.trainable = False
        qnet.summary()
        return qnet, tnet

    def choose_action(self, state, valid_actions, goal, ignore_eps=False):
        if not ignore_eps and np.random.rand()<self.epsilon:
            i = np.random.choice(len(valid_actions))
            return valid_actions[i]
        inputs = merge_input([state], [valid_actions], [goal])
        q = self.qnet.predict(inputs).reshape(-1,)
        i = np.argmax(q)
        return valid_actions[i]

    def learn(self, batch_size):
        state, move, reward, next_state, next_move, goal, done = self.buf.sample(batch_size)
        tinputs = make_qnet_input(next_state, next_move, goal)
        q_hat = self.tnet.predict(tinputs).reshape(-1,)
        target = reward + (1.0-done)*self.gamma*q_hat

        self.qnet.train_on_batch(make_qnet_input(state, move, goal), target)
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
        res = self.model.choose_action(state, move_list, [0]*15)
        if os.path.exists( os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test')):
            # player i [手牌] // [出牌]
            hand_card = []
            for i, n in enumerate(Card.all_card_name):
                hand_card.extend([n]*self.get_hand_card()[i])
            print("DQN Player {}".format(self.player_id), ' ', hand_card, ' // ', Card.visual_card(res))

        return res, None


