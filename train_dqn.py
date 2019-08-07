from game.engine import Game, Agent, Card
from rl_model.dqn import DQNAgent, DQNModel, list_to_mat, state_to_tensor, make_input
from collections import deque
import numpy as np
import tensorflow as tf
import datetime as dt
import os
import argparse
import sys
from rule_utils.rule_based_model import RuleBasedModel
from mcts.mcts_model import MctsModel
from mix_model.mix_model import MixModel

class RandomAgent(Agent):
    def choose(self, state):
        valid_moves = self.get_moves()
        i = np.random.choice(len(valid_moves))
        move = valid_moves[i]
        if os.path.exists("test"):
            # player i [手牌] // [出牌]
            print("RAN Player {}".format(self.player_id), ' ', Card.visual_card(self.get_hand_card()), ' // ', Card.visual_card(move))
        return move, None

MAX_ITERATION = 50
LOAD_EPISODE_PER_ITER = 300
PEAS_EPISODE_PER_ITER = 700
BATCH_SIZE = 64
MIN_BUFSIZE = 3*BATCH_SIZE

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

lord_model = DQNModel( (15,4,11), step_per_update = 40, buf_size=20000, min_epsilon = 0.05, epsilon=0.5)
peas_model = DQNModel( (15,4,11), step_per_update = 40, buf_size=40000, min_epsilon = 0.05, epsilon=0.5)
game = Game([DQNAgent(0, lord_model)] + [RuleBasedModel(i) for i in range(1,3)])
#game = Game([DQNAgent(0, lord_model)] + [DQNAgent(i, peas_model) for i in range(1,3)])
wins0 = deque(maxlen=100)
wins1 = deque(maxlen=100)


def store_transition(pid, lord_win, peas_win, model, state, move, state_, moves_):
    state = state_to_tensor(state)
    move = list_to_mat(move)
    state_ = state_to_tensor(state_)
    done = (lord_win or peas_win)
    if not done:
        move_ = model.choose_action(state_, moves_, ignore_eps=True)
    else:
        move_ = [0]*15
    reward = 0
    move_ = list_to_mat(move_)
    if lord_win:
        reward = 100 if pid==0 else -100
    if peas_win:
        reward = -100 if pid==0 else 100
    model.buf.remember(state, move, reward, state_, move_, done)

def train_lord(it):
    print(dt.datetime.now(),  " Now Training Lord....")
    for i_episode in range(LOAD_EPISODE_PER_ITER):
        _, _, _, mingpai = game.game_reset()
        lord_win, peas_win = False, False
        goal = list_to_mat([0]*15)
        while not (lord_win or peas_win):
            pid, state, _, move, winner0, _ = game.step()
            lord_win = (winner0==0)
            if not lord_win:
                _, _, _, _, winner1, _ = game.step()
                peas_win = (winner1>0)
                if not peas_win:
                    _, _, _, _, winner2, _ = game.step()
                    peas_win = (winner2>0)
            state_, moves_ = game.players[pid].observation()
            store_transition(pid, lord_win, peas_win, lord_model, state, move, state_, moves_)
            if len(lord_model.buf) > MIN_BUFSIZE:
                lord_model.learn(BATCH_SIZE)
        lord_model.buf.update_memory()
        lord_model.update_epsilon(i_episode+it*LOAD_EPISODE_PER_ITER, MAX_ITERATION*LOAD_EPISODE_PER_ITER)
        wins0.append( lord_win==True )
        if i_episode % 100==0:
            print(dt.datetime.now(),  " Episode: {}, lord win rate: {}, eps: {}".format(i_episode, sum(wins0)/len(wins0), lord_model.epsilon) )
        if os.path.exists('test'):
            print(lord_win, ' ', peas_win)
            input()
    lord_model.save("lord_vs_rule.h5")

def train_peas(it):
    print(dt.datetime.now(),  " Now Training Peasant....")
    for i_episode in range(PEAS_EPISODE_PER_ITER):
        _, _, _, mingpai = game.game_reset()
        lord_win, peas_win = False, False
        goal = list_to_mat([0]*15)
        data = [None, [], []] # 位置0的None是占位。1/2位置会用农民的状态与出牌填充。
        while not (lord_win or peas_win):
            _, _, _, _, winner0, _ = game.step()
            lord_win = (winner0==0)
            if not lord_win:
                _, state, moves, move, winner1, _ = game.step()
                peas_win = (winner1>0)
                if len(data[1])>0:
                    store_transition(1, lord_win, peas_win, peas_model, data[1][0], data[1][1], state, moves)
                data[1] = [state, move]
                if peas_win:
                    # 地主下家赢了。说明农民这一步与另一个农民的上一步是正确的。
                    state_, moves_ = game.players[1].observation() # 这个状态理论上不对，但是因为此时done=True，不影响DQN决策。
                    store_transition(1, lord_win, peas_win, peas_model, state, move, state_, moves_)
                    store_transition(2, lord_win, peas_win, peas_model, data[2][0], data[2][1], state_, moves_) #同上
                else:
                    _, state, moves, move, winner2, _ = game.step()
                    peas_win = (winner2>0)
                    if len(data[2])>0:
                        store_transition(1, lord_win, peas_win, peas_model, data[1][0], data[1][1], state, moves)
                    data[2] = [state, move]
                    if peas_win:
                        # 地主上家赢了。说明农民的接牌是正确的。
                        state_, moves_ = game.players[2].observation() # 这个状态理论上不对，但是因为此时done=True，不影响DQN决策。
                        store_transition(1, lord_win, peas_win, peas_model, data[1][0], data[1][1], state_, moves_)
                        store_transition(2, lord_win, peas_win, peas_model, state, move, state_, moves_)
            else:
                # 说明农民最后一手是错的。回溯data，-100
                state_, moves_ = game.players[1].observation() # 这个状态理论上不对，但是因为此时done=True，不影响DQN决策。
                store_transition(1, lord_win, peas_win, peas_model, data[1][0], data[1][1], state_, moves_)
                store_transition(2, lord_win, peas_win, peas_model, data[2][0], data[2][1], state_, moves_)

            if len(peas_model.buf) > MIN_BUFSIZE:
                peas_model.learn(BATCH_SIZE)
        peas_model.buf.update_memory()
        peas_model.update_epsilon(i_episode+it*PEAS_EPISODE_PER_ITER, MAX_ITERATION*PEAS_EPISODE_PER_ITER)
        wins1.append( peas_win==True )
        if i_episode % 100==0:
            print(dt.datetime.now(),  " Episode: {}, peas win rate: {}, eps: {}".format(i_episode, sum(wins1)/len(wins1), peas_model.epsilon) )
        if os.path.exists('test'):
            print(lord_win, ' ', peas_win)
            input()
    peas_model.save("peas.h5")

if __name__=="__main__":
    for it in range(MAX_ITERATION):
        train_lord(it)
        #train_peas(it)

            

