from __future__ import absolute_import

import sys
from engine import Agent, Game, FakeAgent
from gameutil import card_show
import numpy as np


class RandomModel(Agent):
    def choose(self):
        valid_types, valid_moves = self.get_moves(self.game.last_move_type, self.game.last_move)

        # player i [手牌] // [出牌]
        print("Player {}".format(self.player_id), ' ', self.get_hand_card(), end=' // ')

        if len(valid_moves)>0:
            i = np.random.choice(len(valid_types))
            type, move = valid_types[i], valid_moves[i]
            print(move)
        else:
            type = 'yaobuqi'
            move = []
            print("要不起(不要)")

        return type, move



game = Game([RandomModel(i) for i in range(3)])

for i_episode in range(1):
    game.game_reset()
    game.show()
    for i in range(100):
        winner = game.step()
        #game.show()
        if winner != -1:
            print('Winner:{}'.format(winner))
            break
    