import sys
from game.engine import Agent, Game, Card
import numpy as np

class RandomModel(Agent):
    def choose(self, state):
        valid_moves = self.get_moves()
        i = np.random.choice(len(valid_moves))
        move = valid_moves[i]
        # player i [手牌] // [出牌]
        print("RAN Player {}".format(self.player_id), ' ', Card.visual_card(self.get_hand_card()), ' // ', Card.visual_card(move))

        return move, None


if __name__=="__main__":
    game = Game([RandomModel(i) for i in range(3)])

    for i_episode in range(1):
        game.game_reset()
        game.show()
        for i in range(100):
            pid, state, cur_moves, cur_move, winner, info = game.step()
            #game.show()
            if winner != -1:
                print('Winner:{}'.format(winner))
                break
        