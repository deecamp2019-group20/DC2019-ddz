import sys
from pathlib import Path

sys.path.append(str(Path(".")))

from game.engine import *
from game.gameutil import card_show

class RandomModel(Agent):
    
    def choose(self,state):
        return 'dan',[self.cards_left[0]]

game = Game([RandomModel(i) for i in range(3)])

for i_episode in range(1):
    game.game_reset()
    game.show()
    for i in range(100):
        winner = game.step()
        game.show()
        if winner != -1:
            print('Winner:{}'.format(winner))
            break
    