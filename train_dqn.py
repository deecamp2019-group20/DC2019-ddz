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
            pid, state, moves, move, desc, winner = game.step()

            if len(record)==len(game.players):
                p = record.popleft()
                game.players[pid].store_transition(...)
            record[pid] = (pid, state, move, desc)

            if winner!=-1:
                game.players[pid].store_transition(...)
                while len(record)>0:
                    p = record.pop()
                    game.players[p.player_id].store_transition(...)
            game.players[pid].learn()

            

