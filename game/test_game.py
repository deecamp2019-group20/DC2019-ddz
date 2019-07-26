import sys
from pathlib import Path
sys.path.append(str(Path(".")))

from game.engine import *
from game.gameutil import card_show
import numpy as np


class RandomModel(Agent):
    def choose(self, state):
        valid_types, valid_moves = self.get_moves(state.last_move_type, state.last_move)

        # player i [手牌] // [出牌]
        print("Player {}".format(self.player_id), end=' [')
        for i in range(len(self.cards_left)):
            print(self.cards_left[i], end= ', ' if i!=len(self.cards_left)-1 else '')
        print(']', end=' // ')

        if len(valid_moves)>0:
            i = np.random.choice(len(valid_types))
            type, move = valid_types[i], valid_moves[i]
            print('[', end='')
            for i in range(len(move)):
                print(move[i], end= ', ' if i!=len(move)-1 else '')
            print(']')
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
    