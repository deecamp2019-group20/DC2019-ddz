from game.engine import Game, Agent, Card
from rl_model.dqn import DQNAgent, DQNModel, list_to_mat, state_to_tensor, make_input
from collections import deque
import numpy as np
import tensorflow as tf
import datetime as dt
import os

MAX_EPISODE = 50000
BATCH_SIZE = 64
MIN_BUFSIZE = 3*BATCH_SIZE

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

class RandomModel(Agent):
    def choose(self, state):
        valid_moves = self.get_moves()
        i = np.random.choice(len(valid_moves))
        move = valid_moves[i]
        if os.path.exists("test"):
            # player i [手牌] // [出牌]
            print("RAN Player {}".format(self.player_id), ' ', Card.visual_card(self.get_hand_card()), ' // ', Card.visual_card(move))

        return move, None

def train_lord():
    rl_model = DQNModel( (15,4,8), step_per_update = 40, buf_size=40000, min_epsilon = 0.05, epsilon=0.5)
    game = Game([DQNAgent(0, rl_model)] + [RandomModel(i) for i in range(1,3)])
    wins = deque(maxlen=100)

    for i_episode in range(MAX_EPISODE):
        _, _, _, mingpai = game.game_reset()
        #game.show()
        lord_win, pes_win = False, False
        goal = list_to_mat([0]*15)
        while not (lord_win or pes_win):
            pid, state, _, move, winner0, _ = game.step()
            lord_win = (winner0==0)
            if not lord_win:
                _, _, _, _, winner1, _ = game.step()
                pes_win = (winner1>0)
                if not (lord_win or pes_win):
                    _, _, _, _, winner2, _ = game.step()
                    pes_win |= (winner2>0)
            state_, moves_ = game.players[pid].observation()

            #prepare data:
            state = state_to_tensor(state)
            move = list_to_mat(move)
            state_ = state_to_tensor(state_)
            done = (lord_win or pes_win)
            if not done:
                move_ = rl_model.choose_action(state_, moves_, ignore_eps=True)
            else:
                move_ = [0]*15
            reward = 0
            move_ = list_to_mat(move_)
            if lord_win:
                reward = 100
            if pes_win:
                reward = -100


            rl_model.buf.remember(state, move, reward, state_, move_, done)
            if len(rl_model.buf) > MIN_BUFSIZE:
                rl_model.learn(BATCH_SIZE)
        rl_model.buf.update_memory()
        rl_model.update_epsilon(i_episode, MAX_EPISODE)
        wins.append( lord_win==True )
        if i_episode % 100==0:
            print(dt.datetime.now(),  " Episode: {}, lord win rate: {}, eps: {}".format(i_episode, sum(wins)/len(wins), rl_model.epsilon) )
        if i_episode % 500==0:
            rl_model.save()
        if os.path.exists('test'):
            print(lord_win, ' ', pes_win)
            input()

if __name__=="__main__":
    train_lord()

            

