### python文件介绍
- actions.py 所有的action
- engine.py 引擎相关类
- gameutil.py 相关工具类

### 使用方法示例
```Python
from.game.engine import Agent

# 自己的模型继承Agent类并重写choose方法
class MyModel(Agent):
    
    def choose(self,state):
		'''
			模型根据当前状态选择出牌的方法

            参数说明：
			state.player_id 当前玩家的id，0代表地主，1代表地主下家，2代表地主上家
			self.get_hand_card() 当前剩余手牌
			self.game.cards_out 所有玩家按序打出的牌。格式：[ (player_id, type, move), ... ]
			self.last_move_type 上家出的牌的类型
			self.last_move = last_move 上家出的牌


            返回值说明：
			返回模型选择的当前应打出的牌及其类型
			牌的类型有["dan", "dui", "san", "san_dai_yi", "san_dai_er", "shunzi"]，后期可扩展
			打出的牌为一个list，元素为Card类
		'''

        return 'dan', [self.cards_left[0]]

game = Game([MyModel(i) for i in range(3)])
MAX_ROUNDS = 100
TRAIND_ID = 0	# 进行训练的模型，0代表地主，1代表地主下家，2代表地主上家

for i_episode in range(1):
    game.game_reset()
    game.show()	# 输出当前各个玩家的手牌
    for i in range(MAX_ROUNDS):
        winner = game.step()	#-1代表游戏未结束，0代表地主获胜，1代表地主下家获胜，2代表地主上家获胜
        game.show()
        if winner != -1:
            
            if TRAIND_ID == 0 and winner == 0:
                # do some positive reward
            elif TRAIND_ID != 0 and winner != 0:
                # do some positive reward
            else:
                # do some negative reward
            
            break
    
```