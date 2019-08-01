from game.engine import Game
from rl.dqn import DQNAgent, build_model
from test_game import RandomModel
from collections import deque

def is_winner(pid, win_flag):
    if win_flag<0:
        return False
    if pid==0:
        return win_flag==0
    return True

if __name__ == "__main__":
    rl_model =  build_model()
    game = Game([DQNAgent(0, rl_model)] + [RandomModel(i) for i in range(1,4)])

    for i_episode in range(1):
        _, _, _, mingpai = game.game_reset()
        game.show()
        record = deque()
        for i in range(100):
            pid, state, moves, move, winner, info = game.step()



            

