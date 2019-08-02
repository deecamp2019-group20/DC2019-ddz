from game.engine import Game, Agent
from rl.dqn import DQNAgent, DQNModel
from collections import deque
import numpy as np

MAX_EPISODE = 10000
BATCH_SIZE = 256
MIN_BUFSIZE = 3*BATCH_SIZE

class RandomModel(Agent):
    def choose(self, state):
        valid_moves = self.get_moves()
        i = np.random.choice(len(valid_moves))
        move = valid_moves[i]
        return move, None

def train_lord():
    rl_model = DQNModel()
    game = Game([DQNAgent(0, rl_model)] + [RandomModel(i) for i in range(1,3)])
    wins = deque(maxlen=100)

    for i_episode in range(MAX_EPISODE):
        _, _, _, mingpai = game.game_reset()
        #game.show()
        lord_win, pes_win = False, False
        while not (lord_win or pes_win):
            pid, state, moves, move, winner0, _ = game.step()
            lord_win = (winner0==0)
            if not lord_win:
                _, _, _, _, winner1, _ = game.step()
                pes_win = (winner1>0)
                if not (lord_win or pes_win):
                    _, _, _, _, winner2, _ = game.step()
                    pes_win |= (winner2>0)
            state_, moves_ = game.players[pid].observation()
            reward = 0
            if lord_win:
                reward = 1
            if pes_win:
                reward = -1
            rl_model.store_transition(state, moves, move, reward, state_, moves_)
            if len(rl_model.buf) > MIN_BUFSIZE:
                rl_model.learn(BATCH_SIZE)
        wins.append( lord_win==True )
        if i_episode % 100==0:
            print("Episode: {}, lord win rate: {}".format(i_episode, sum(wins)/len(wins)) )


if __name__=="__main__":
    train_lord()

            

