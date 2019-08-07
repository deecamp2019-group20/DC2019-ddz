from game.engine import Game, Agent, Card
from rl_model.dqn import DQNAgent, DQNModel, list_to_mat, state_to_tensor, make_input
from rule_utils.rule_based_model import RuleBasedModel
from mcts.mcts_model import MctsModel
from mix_model.mix_model import MixModel

from collections import deque
import numpy as np
import tensorflow as tf
import datetime as dt
import os

class RandomAgent(Agent):
    def choose(self, state):
        valid_moves = self.get_moves()
        i = np.random.choice(len(valid_moves))
        move = valid_moves[i]
        if os.path.exists("test"):
            # player i [手牌] // [出牌]
            print("RAN Player {}".format(self.player_id), ' ', Card.visual_card(self.get_hand_card()), ' // ', Card.visual_card(move))
        return move, None

MAX_EPISODE = 1000

lord_model = DQNModel( (15,4,11), step_per_update = 40, buf_size=20000, min_epsilon = 0.05, epsilon=0.5)
game = Game([DQNAgent(0, lord_model)] + [RandomAgent(i) for i in range(1,3)])
wins0 = deque(maxlen=100)

def inference():
    for i_episode in range(MAX_EPISODE):
        game.game_reset()
        lord_win, peas_win = False, False
        goal = list_to_mat([0]*15)
        while not (lord_win or peas_win):
            _, _, _, _, winner0, _ = game.step()
            lord_win = (winner0==0)
            peas_win = (winner0>0)
        wins0.append( lord_win==True )
        print(dt.datetime.now(),  " Episode: {}, lord win rate: {}, eps: {}".format(i_episode, sum(wins0)/len(wins0), lord_model.epsilon) )
        print(lord_win, ' ', peas_win)
        #input()


if __name__=="__main__":
    lord_model.load("lord_tmp.h5")
    lord_model.epsilon = 0.05
    inference()
