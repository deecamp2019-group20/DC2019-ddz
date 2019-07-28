import random
from collections import deque
import numpy as np


class ReplayerBuffer():
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


class ReplayerBufferHER(ReplayerBuffer):
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


###############
# Sample Code #
###############

import tensorflow as tf
import time

class BitFlip():
    def __init__(self, n, reward_type):
        self.n = n # number of bits
        self.reward_type = reward_type
    def reset(self):
        self.goal = np.random.randint(2, size=(self.n)) # a random sequence of 0's and 1's
        self.state = np.random.randint(2, size=(self.n)) # another random sequence of 0's and 1's as initial state
        return np.copy(self.state), np.copy(self.goal)
    def step(self, action):
        self.state[action] = 1-self.state[action] # flip this bit
        done = np.array_equal(self.state, self.goal)
        if self.reward_type == 'sparse':
            reward = 0 if done else -1
        else:
            reward = -np.sum(np.square(self.state-self.goal))
        return np.copy(self.state), reward, done
    def render(self):
        print("\rstate :", np.array_str(self.state), end=' '*10)

class DQNAgent():
    def __init__(self, state_size, action_size, goal_size, use_double_dqn=True, clip_target_value=True):
        self.state_size = state_size
        self.goal_size = goal_size
        self.action_size = action_size
        self.use_double_dqn = use_double_dqn
        self.clip_target_value = clip_target_value
        self.memory = ReplayerBuffer()
        self.epsilon = 0.2  # exploration
        self.epsilon_min = 0.02  # min exploration
        self.epsilon_decay = 0.95
        self.tau = 0.95  # target net update weight
        self.gamma = 0.98
        self.batch_size = 128
        self.learning_rate = 1e-3
        self.buffer_size = int(1e6)
        self._set_model()

    def _set_model(self):  # set value network
        tf.reset_default_graph()
        self.sess = tf.Session()

        self.tfs = tf.placeholder(tf.float32, [None, self.state_size], 'state')
        self.tfs_ = tf.placeholder(tf.float32, [None, self.state_size], 'next_state')
        self.tfg = tf.placeholder(tf.float32, [None, self.goal_size], 'goal')
        self.tfa = tf.placeholder(tf.int32, [None, ], 'action')
        self.tfr = tf.placeholder(tf.float32, [None, ], 'reward')
        self.tfd = tf.placeholder(tf.float32, [None, ], 'done')

        def _build_qnet(state, scope, trainable, reuse):
            with tf.variable_scope(scope, reuse=reuse):
                net = tf.layers.dense(tf.concat([state, self.tfg], axis=1), 256, activation=tf.nn.relu,
                                      trainable=trainable)
                q = tf.layers.dense(net, self.action_size, trainable=trainable)
            return q, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

        self.q_eval, e_params = _build_qnet(self.tfs, 'eval', trainable=True, reuse=False)
        self.q_targ, t_params = _build_qnet(self.tfs_, 'target', trainable=False, reuse=False)

        self.update_op = [tf.assign(t, self.tau * e + (1 - self.tau) * t) for t, e in
                          zip(t_params, e_params)]  # soft update

        if self.use_double_dqn:
            q_eval_next, _ = _build_qnet(self.tfs_, 'eval', trainable=True, reuse=True)  # reuse the same eval net
            q_eval_next_best_action = tf.argmax(q_eval_next, 1)
            self.q_target_value = tf.reduce_sum(self.q_targ * tf.one_hot(q_eval_next_best_action, self.action_size),
                                                axis=1)
        else:
            self.q_target_value = tf.reduce_max(self.q_targ, axis=1)

        self.q_target_value = self.tfr + self.gamma * (1 - self.tfd) * self.q_target_value

        if self.clip_target_value:
            self.q_target_value = tf.clip_by_value(self.q_target_value, -1 / (1 - self.gamma), 0)

        self.q_eval_action_value = tf.reduce_sum(self.q_eval * tf.one_hot(self.tfa, self.action_size), axis=1)

        self.loss = tf.losses.mean_squared_error(self.q_target_value, self.q_eval_action_value)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.saver = tf.train.Saver()

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, state, goal):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        act_values = self.sess.run(self.q_eval, {self.tfs: state, self.tfg: goal})
        return np.argmax(act_values[0])  # use tf.argmax is much slower, so use np

    def replay(self, optimization_steps):
        if len(self.memory) < self.batch_size:  # if there's no enough transitions, do nothing
            return 0

        losses = 0
        for _ in range(optimization_steps):
            states, actions, rewards, next_states, dones, goals = self.memory.sample(self.batch_size)

            loss, _ = self.sess.run([self.loss, self.train_op],
                                    {self.tfs: states, self.tfg: goals, self.tfa: actions,
                                     self.tfr: rewards, self.tfs_: next_states, self.tfd: dones})
            losses += loss

        return losses / optimization_steps  # return mean loss

    def update_target_net(self, decay=True):
        self.sess.run(self.update_op)
        if decay:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

if __name__=="__main__":
    size = 30
    env = BitFlip(size, 'sparse')
    agent = DQNAgent(size, size, size)

    use_her = True  # use hindsight experience replay or not
    num_epochs = 250
    num_episodes = 16
    optimization_steps = 40
    K = 4  # number of random future states

    losses = []
    success_rate = []

    start = time.clock()
    for i in range(num_epochs):
        successes = 0
        for n in range(num_episodes):
            state, goal = env.reset()
            for t in range(size):
                action = agent.choose_action([state], [goal])
                next_state, reward, done = env.step(action)
                agent.memory.remember(state, action, reward, next_state, done, goal)

                state = next_state
                if done:
                    break
            successes += done

            agent.memory.update_memory()


        mean_loss = agent.replay(optimization_steps)
        agent.update_target_net()

        losses.append(mean_loss)
        success_rate.append(successes / num_episodes)
        print("\repoch", i + 1, "success rate", success_rate[-1], 'loss %.2f' % losses[-1],
              'exploration %.2f' % agent.epsilon, end=' ' * 10)

    print("Training time : %.2f" % (time.clock() - start), "s")