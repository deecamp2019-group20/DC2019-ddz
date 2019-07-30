from __future__ import absolute_import

import sys
from engine import Agent, Game, ManualAgent, Card
import numpy as np


class RandomModel(Agent):
    def choose(self, state):
        valid_moves = self.get_moves(self.game.last_move, self.game.last_desc)

        # player i [手牌] // [出牌]
        hand_card = []
        for ls in self.get_hand_card().values():
            hand_card.extend(list(ls))
        print("Player {}".format(self.player_id), ' ', hand_card, end=' // ')

        i = np.random.choice(len(valid_moves))
        tmp = valid_moves[i]
        move = []
        for k in Card.all_card_name:
            move.extend([int(k)]* tmp.get(k, 0))
        print(move)

        return move



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
    