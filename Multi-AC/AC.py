"""
Actor Critic (A3C) with discrete action space
For Deecamp_2019 Group 20's Project -- Dou Di Zhu AI
Tianhao Zhang
C.T.R.L
2019.7.27

Using:
tensorflow 1.11.0
python 3.6.7
"""

import tensorflow as tf
import numpy as np

import os
from game.engine import Agent, Game, ManualAgent, Card, group_by_type
from game.card_util import type_encoding, inv_type_encoding, name_to_rank
from tqdm import tqdm

# np.random.seed(2)
# tf.set_random_seed(2)  # reproducible

#   Parameters
GPU = False
OUTPUT_GRAPH = False
load_model = True
MAX_EPISODES = 100
MAX_EPISODE = 100
# renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000   # maximum time step in one episode
RENDER = True  # rendering wastes time
GAMMA = 0.95     # reward discount in TD error
LR_A = 0.0001    # learning rate for actor
LR_C = 0.0001     # learning rate for critic

#   Functions

# 联合动作


def ComAction(a1, a2=None, a3=None, a4=None):
    #           type_encoding = {'buyao':0, 'dan':1, 'dui':2, 'san':3, 'san_yi':4, 'san_er':5, \
    #                 'dan_shun':6, 'er_shun':7, 'feiji':8, 'xfeiji':9, 'dfeiji':10, \
    #                 'zha':11, 'si_erdan':12, 'si_erdui':13, 'wangzha':14}
    # [不要 单 双 三 炸弹 王炸// 单顺 双顺 三顺  三带一 三带一对 // 双飞机带两单 双飞机带两双 四带两单 四带两双//]

    t = [0] * 15
    if a1 == 1:
        t[a2] = 1
    if a1 == 2:
        t[a2] = 2
    if a1 == 3:
        t[a2] = 3
    if a1 == 4:
        t[a2] = 4
    if a1 == 5:
        t[-1] = 1
        t[-2] = 1
    if a1 == 6:
        m = a3 - a2
        for i in range(m + 1):
            t[i + a2] = 1
    if a1 == 7:
        m = a3 - a2
        for i in range(m + 1):
            t[i + a2] = 2
    if a1 == 8:
        m = a3 - a2
        for i in range(m + 1):
            t[i + a2] = 3
    if a1 == 9:
        t[a2] = 3
        t[a3] = 1
    if a1 == 10:
        t[a2] = 3
        t[a3] = 2
    if a1 == 11:
        t[a2] = 3
        t[a2+1] = 3
        t[a3] = 1
        t[a4] = 1
    if a1 == 12:
        t[a2] = 3
        t[a2+1] = 3
        t[a3] = 2
        t[a4] = 2
    if a1 == 13:
        t[a2] = 4
        t[a3] = 1
        t[a4] = 1
    if a1 == 14:
        t[a2] = 4
        t[a3] = 2
        t[a4] = 2
    return t

# 列表转矩阵


def list_to_array(Ls):
    Array = np.zeros(shape=(15, 4), dtype=int)
    for index, value in enumerate(Ls):
        if value == 0:
            continue
        else:
            for rndex in range(value):
                Array[index, rndex] = 1
    return Array

# 状态转矩阵


def state_to_array(s, mingpai):
        # 手牌
    hand = list_to_array(s.hand)
    # 地主公开的牌
    mingpai = list_to_array(mingpai)
    # 上上次打的牌
    down_move = list_to_array(s.last_last_move_)
    # 上次打的牌
    up_move = list_to_array(s.last_move_)
    # 下家打过的牌
    downOut = list_to_array(s.down_out)
    # 上家打过的牌
    upOut = list_to_array(s.up_out)
    # 桌面的牌
    zOut = list_to_array(s.out)
    # 记牌器牌
    jipaiqi = list_to_array(s.other_hand)

    A = np.array([hand, mingpai, down_move, up_move,
                  downOut, upOut, zOut, jipaiqi])
    B = np.moveaxis(A, 0, -1)
    B = B.astype(np.float32)
    B = B[np.newaxis, :]
    return B

#   Network


class Actor(object):
    def __init__(self, sess, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(
            tf.float32, [None, 15, 4, 8], name='state')
        self.s1 = tf.placeholder(tf.float32, 15, name='state_1')
        self.s2 = tf.placeholder(tf.float32, 30, name='state_2')
        self.s3 = tf.placeholder(tf.float32, 30, name='state_3')
        self.s4 = tf.placeholder(tf.float32, 30, name='state_4')
        self.a1 = tf.placeholder(tf.int32, None, "act_1")
        self.a2 = tf.placeholder(tf.int32, None, "act_2")
        self.a3 = tf.placeholder(tf.int32, None, "act_3")
        self.a4 = tf.placeholder(tf.int32, None, "act_4")
        self.td_error = tf.placeholder(
            tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            l1 = tf.layers.conv2d(self.s, 64, 3, 1, "SAME",
                                  activation=tf.nn.leaky_relu, name='la1')
            l2 = tf.layers.conv2d(l1, 128, 3, 1, "SAME",
                                  activation=tf.nn.leaky_relu, name='la2')
            l3 = tf.layers.max_pooling2d(
                l2, [2, 2], [1, 1], "SAME", name='la3')
            l4 = tf.layers.flatten(l3, name='la4')
            l5 = tf.layers.dense(
                inputs=l4,
                units=256,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(
                    0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='la5'
            )
            # l6 = tf.concat(1, [tf.expand_dims(l5,1),tf.expand_dims(self.s1,1)], name = 'la6')

            l6 = tf.concat([l5, tf.expand_dims(self.s1, 0)], 1, name='la6')
            # 第一个网络输出 选牌类型
            # [不要 单 双 三 炸弹 王炸 // 单顺 双顺 三顺 // 三带一 三带一对 // 双飞机带两单 双飞机带两双 四带两单 四带两双//]
            self.acts1_prob = tf.layers.dense(
                inputs=l6,
                units=15,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(
                    0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts1_prob'
            )
            l7 = tf.layers.dense(
                inputs=l4,
                units=256,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(
                    0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='la7'
            )
            l8 = tf.concat([l6, l7, tf.expand_dims(self.s2, 0)], 1, name='la8')
            #   第二个网络选择合法的出牌数字 3 - 大王
            self.acts2_prob = tf.layers.dense(
                inputs=l8,
                units=15,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(
                    0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts2_prob'
            )
            l9 = tf.layers.dense(
                inputs=l4,
                units=256,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(
                    0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='la9'
            )
            l10 = tf.concat(
                [l8, l9, tf.expand_dims(self.s3, 0)], 1, name='la10')
            #   第三个网络选择合法的出牌数字 3 - 大王
            self.acts3_prob = tf.layers.dense(
                inputs=l10,
                units=15,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(
                    0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts3_prob'
            )
            l11 = tf.layers.dense(
                inputs=l4,
                units=256,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(
                    0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='la11'
            )
            l12 = tf.concat(
                [l10, l11, tf.expand_dims(self.s4, 0)], 1, name='la12')
            #   第四个网络选择合法的出牌数字 3 - 大王
            self.acts4_prob = tf.layers.dense(
                inputs=l12,
                units=15,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(
                    0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts4_prob'
            )

        with tf.variable_scope('exp_v'):
            log1_prob = tf.log(self.acts1_prob[0, self.a1] + 1e-3)
            log2_prob = tf.log(self.acts2_prob[0, self.a2] + 1e-3)
            log3_prob = tf.log(self.acts3_prob[0, self.a3] + 1e-3)
            log4_prob = tf.log(self.acts4_prob[0, self.a4] + 1e-3)
            # advantage (TD_error) guided loss
            self.exp1_v = tf.reduce_mean(log1_prob * self.td_error)
            self.exp2_v = tf.reduce_mean(log2_prob * self.td_error)
            self.exp3_v = tf.reduce_mean(log3_prob * self.td_error)
            self.exp4_v = tf.reduce_mean(log4_prob * self.td_error)

        with tf.variable_scope('train'):
            # minimize(-exp_v) = maximize(exp_v)
            self.train1_op = tf.train.AdamOptimizer(lr).minimize(-self.exp1_v)
            self.train2_op = tf.train.AdamOptimizer(
                lr).minimize(-self.exp1_v-self.exp2_v)
            self.train3_op = tf.train.AdamOptimizer(
                lr).minimize(-self.exp1_v-self.exp2_v-self.exp3_v)
            self.train4_op = tf.train.AdamOptimizer(
                lr).minimize(-self.exp1_v-self.exp2_v-self.exp3_v-self.exp4_v)

    def learn1(self, s, s1, a1, td):
        feed_dict = {self.s: s, self.s1: s1,
                     self.a1: a1, self.td_error: td, self.s1: s1}
        self.sess.run([self.train1_op, self.exp1_v], feed_dict)

    def learn2(self, s, s1, s2, a1, a2, td):
        feed_dict = {self.s: s, self.a1: a1, self.a2: a2,
                     self.td_error: td, self.s1: s1, self.s2: s2}
        self.sess.run([self.train2_op, self.exp1_v, self.exp2_v], feed_dict)

    def learn3(self, s, s1, s2, s3, a1, a2, a3, td):
        feed_dict = {self.s: s, self.a1: a1, self.a2: a2, self.a3: a3, self.td_error: td,
                     self.s1: s1, self.s2: s2, self.s3: s3}
        self.sess.run(
            [self.train3_op, self.exp1_v, self.exp2_v, self.exp3_v], feed_dict)

    def learn4(self, s, s1, s2, s3, s4, a1, a2, a3, a4, td):
        feed_dict = {self.s: s, self.a1: a1, self.a2: a2, self.a3: a3, self.a4: a4, self.td_error: td,
                     self.s1: s1, self.s2: s2, self.s3: s3, self.s4: s4}
        self.sess.run(
            [self.train4_op, self.exp1_v, self.exp2_v, self.exp3_v, self.exp4_v], feed_dict)

    def choose_action1(self, s, s1):
        # get probabilities for all actions
        probs = self.sess.run(self.acts1_prob, {self.s: s, self.s1: s1})
        s1 = np.array(s1).reshape(1, 15)
        probs = probs * s1
        # return a int
        sum_ = np.sum(probs)
        probs = probs/sum_
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())

    def choose_action2(self, s, s1, s2):
        # get probabilities for all actions
        probs = self.sess.run(
            self.acts2_prob, {self.s: s, self.s1: s1, self.s2: s2})
        s2 = np.array(s2[:15]).reshape(1, 15)
        probs = probs * s2
        # return a int
        sum_ = np.sum(probs)
        probs = probs/sum_
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())

    def choose_action3(self, s, s1, s2, s3):
        # get probabilities for all actions
        probs = self.sess.run(
            self.acts3_prob, {self.s: s, self.s1: s1, self.s2: s2, self.s3: s3})
        s3 = np.array(s3[:15]).reshape(1, 15)
        probs = probs * s3
        sum_ = np.sum(probs)
        probs = probs/sum_
        # return a int
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())

    def choose_action4(self, s, s1, s2, s3, s4):
        # get probabilities for all actions
        probs = self.sess.run(self.acts4_prob, {
                              self.s: s, self.s1: s1, self.s2: s2, self.s3: s3, self.s4: s4})
        s4 = np.array(s4[:15]).reshape(1, 15)
        probs = probs * s4
        sum_ = np.sum(probs)
        probs = probs/sum_
        # return a int
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())


class Critic(object):
    def __init__(self, sess, lr=0.01):
        self.sess = sess
        self.s = tf.placeholder(
            tf.float32, [None, 15, 4, 8], name='state')
        self.s1 = tf.placeholder(tf.float32, 15, name='state_1')
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.conv2d(self.s, 64, 3, 1, "SAME",
                                  activation=tf.nn.leaky_relu, name='lc1')
            l2 = tf.layers.conv2d(l1, 128, 3, 1, "SAME",
                                  activation=tf.nn.leaky_relu, name='lc2')
            l3 = tf.layers.max_pooling2d(
                l2, [2, 2], [1, 1], "SAME", name='lc3')
            l4 = tf.layers.flatten(l3, name='lc4')
            l5 = tf.layers.dense(
                inputs=l4,
                units=256,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(
                    0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='la5'
            )
            l6 = tf.concat([l5, tf.expand_dims(self.s1, 0)], 1, name='lc6')

            l7 = tf.layers.dense(
                inputs=l6,
                units=20,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(
                    0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l7,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(
                    0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            # TD_error = (r+gamma*V_next) - V_eval
            self.loss = tf.square(self.td_error)
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_, s1):

        v_ = self.sess.run(self.v, {self.s: s_, self.s1: s1})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                    {self.s: s, self.v_: v_, self.r: r, self.s1: s1})
        return td_error


class RandomModel(Agent):
    def choose(self, state):
        valid_moves = self.get_moves(self.game.last_move, self.game.last_desc)

        # # player i [手牌] // [出牌]
        # hand_card = []
        # for ls in self.get_hand_card().values():
        #     hand_card.extend(list(ls))
        # print("Player {}".format(self.player_id), ' ', hand_card, end=' // ')

        i = np.random.choice(len(valid_moves))
        tmp = valid_moves[i]
        move = []
        for k in Card.all_card_name:
            move.extend([int(k)] * tmp.get(k, 0))
        # print(move)

        return move, None


class MyModel(Agent):
    def choose(self, state):
        a1, a2, a3, a4 = None, None, None, None
        s1, s2, s3, s4 = [], [], [], []
        state, move = self.observation()
        s = state_to_array(state, mingpai)
        moves = group_by_type(move)
        realMoveType = list(moves.keys())
        realType = [0] * 15
        for types in realMoveType:
            realType[type_encoding[types]] = 1
        s1 = realType
        a1 = actor.choose_action1(s, s1)
        if a1 != 0:
            key = inv_type_encoding[a1]
            realType2 = [0] * 15
            for i in list(set(moves[key])):
                realType2[i] = 1
            # s2 = env.step(a1)
            # a是代表牌型，环境返回该牌型下可以打的数字为s2 = env.xxx()
            s2 = realType2 + realType
            sa1 = [0] * 15
            sa1[a1] = 1
            # 传 a1 和 s2
            a2 = actor.choose_action2(s, s1, s2)
            if a1 > 5:
                t = [0] * 15
                for m in move:
                    if m['type'] == key:
                        if m['main'] == a2 + 1:
                            for i in eval(m['kicker']):
                                t[name_to_rank[i] - 1] = 1

                # s3 = env.step(a2)
                s3 = t + realType
                a3 = actor.choose_action3(s, s1, s2, s3)
                if a1 > 10:
                    s4 = t
                    s4[a3] = 0
                    s4 = s4 + realType
                    #  s4 = env.step(a2)
                    a4 = actor.choose_action4(s, s1, s2, s3, s4)
        # 根据a1 a2 a3 a4 重构动作牌 比如 333+6
        a = ComAction(a1, a2, a3, a4)
        # s_, r, done = env.step(a)
        move_information = [s, s1, s2, s3, s4, a1, a2, a3, a4]
        a_move = []
        for i, n in enumerate(Card.all_card_name):
            a_move.extend([int(n)]*a[i])

        # # show
        # # player i [手牌] // [出牌]
        # hand_card = []
        # for ls in self.get_hand_card().values():
        #     hand_card.extend(list(ls))
        # print("Player {}".format(self.player_id), ' ', hand_card, end=' // ')
        # print(a_move)
        return a_move, move_information


#   Environments
env = Game([MyModel(0), RandomModel(1), RandomModel(2)])

#   Main
if __name__ == "__main__":
    #   GPU/CPU
    if GPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # IF MULTI GPUS = '0,1'
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        sess = tf.Session()

    actor = Actor(sess, lr=LR_A)
    # we need a good teacher, so the teacher should learn faster than the actor
    critic = Critic(sess, lr=LR_C)
    saver = tf.train.Saver(max_to_keep=5)
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state("./drqn")
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        print('Successfully loading Model!')
    else:
        sess.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        tf.summary.FileWriter("logs/", sess.graph)

    for I_episode in tqdm(range(MAX_EPISODES)):
        rate = []

        for i_episode in range(MAX_EPISODE):
            # for i_episode in range(MAX_EPISODE):
            _, _, _, mingpai = env.game_reset()
            win = -1
            done = -1

            # pid player id; state 一个状态类; cur_moves 所有可以打的类型; cur_move 我打了哪个
            # 手牌
            # print(a)
            # s = s.astype(np.float32)
            # s = s[np.newaxis, :]
            t = 0
            track_r = []
            while True:
                # [PASS // 单 双 三 炸弹 王炸 // 单顺 双顺 三顺  三带一 三带一对 // 双飞机带两单 双飞机带两双 四带两单 四带两双//]
                # 环境告诉我当前可以打的手牌为 s1 = env.show()  #把state改掉 返回一个array
                # 人员，执行前的状态，可选动作集，选了哪个动作，动作的描述，有没有人赢
                player_id, pre_state, _, move, _, win, information = env.step()
                if win == -1:
                    _, _, _, _, _, done, _ = env.step()
                    if done == -1:
                        _, _, _, _, _, done, _ = env.step()
                # s = state_to_array(pre_state, mingpai)
                j, _ = env.players[0].observation()
                s_ = state_to_array(j, mingpai)
                s, s1, s2, s3, s4, a1, a2, a3, a4 = information
                a = Card.vectorized_card_list(move)

                if win == 0:
                    r = 100
                    rate.append(1)
                if win == -1 and done == -1:
                    r = 0
                if win == -1 and done > -1:
                    r = -100
                    rate.append(0)

                track_r.append(r)

                # gradient = grad[r + gamma * V(s_) - V(s)]
                # ss为出的牌
                td_error = critic.learn(s, r, s_, a)
                # true_gradient = grad[logPi(s,a) * td_error]
                if a1 == 0:
                    actor.learn1(s, s1, a1, td_error)
                if 0 < a1 < 6:
                    actor.learn2(s, s1, s2, a1, a2, td_error)
                if 5 < a1 < 11:
                    actor.learn3(s, s1, s2, s3, a1, a2, a3, td_error)
                if a1 > 10:
                    actor.learn4(s, s1, s2, s3, s4, a1, a2, a3, a4, td_error)

                s = s_
                t += 1
                # print("t:")
                # print(t)

                if win != -1 or t >= MAX_EP_STEPS or done > 0:
                    ep_rs_sum = sum(track_r)

                    if 'running_reward' not in globals():
                        running_reward = ep_rs_sum
                    else:
                        running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
                    # print("episode:", i_episode, "  reward:", int(ep_rs_sum))
                    # print(sum(rate)/len(rate))
                    break
        saver.save(sess, './drqn/model-'+str(I_episode)+'.cptk')
        rrate = sum(rate)/len(rate)
        print('Episode: ', I_episode, "  winning_rate: ", rrate)
