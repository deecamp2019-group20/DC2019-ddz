# -*- coding: utf-8 -*-
"""
自定义相关类
"""

from .gameutil import card_show
import numpy as np
from typing import List, Tuple

############################################
#                 游戏类                   #
############################################                   
class Game(object):
    def __init__(self, agents: List['Agent']):
        # 初始化players
        self.players = []
        for p in agents:
            p.set_game(self)
            self.players.append(p)
        self.game_reset()
        
    # 游戏环境重置
    def game_reset(self):
        #初始化一副扑克牌类
        cards = Cards()
        #洗牌
        np.random.shuffle(cards.cards)
        #发牌并排序
        p1_cards = cards.cards[:20]
        p1_cards.sort(key=lambda x: x.rank)
        p2_cards = cards.cards[20:37]
        p2_cards.sort(key=lambda x: x.rank)
        p3_cards = cards.cards[37:]
        p3_cards.sort(key=lambda x: x.rank)
        self.players[0].cards_left = p1_cards
        self.players[1].cards_left = p2_cards
        self.players[2].cards_left = p3_cards
        self.cards_out = []

        #play相关参数
        self.end = False    # 游戏是否结束
        self.last_move_type = self.last_move = "start"
        self.playround = 1  # 回合数
        self.index = 0  # 当前玩家的id，0代表地主，1代表地主下家，2代表地主上家
        self.yaobuqis = []
    
    #游戏进行    
    def step(self):
        state = State(self.index, self.players[self.index].cards_left, self.cards_out, self.last_move_type, self.last_move)
        self.last_move_type, self.last_move, self.end, self.yaobuqi = self.players[self.index].step(state)
        if self.yaobuqi:
            self.yaobuqis.append(self.index)
        else:
            self.yaobuqis = []
        #都要不起
        if len(self.yaobuqis) == len(self.players)-1:
            self.yaobuqis = []
            self.last_move_type = self.last_move = "start"

        winner = -1
        if self.end:
            winner = self.index

        self.index = self.index + 1
        #一轮结束
        if self.index >= len(self.players):
            #playrecords.show("=============Round " + str(playround) + " End=============")
            self.playround = self.playround + 1
            #playrecords.show("=============Round " + str(playround) + " Start=============")
            self.index = 0  

        return winner

    def show(self):
        for i in range(3):
            card_show(self.players[i].cards_left,"Player {}".format(i),1)

############################################
#              扑克牌相关类                 #
############################################

class Card(object):
    """
    扑克牌类
    """
    def __init__(self, card_type):
        self.card_type = card_type  # '牌面数字-花色-大小排名' 举例来说，红桃A的card_type为'1-♥-12'
        self.name = self.card_type.split('-')[0] # 名称,即牌面数字
        self.color = self.card_type.split('-')[1] # 花色
        # 大小
        self.rank = int(self.card_type.split('-')[2])

    #判断大小
    def bigger_than(self, card_instance):
        if (self.rank > card_instance.rank):
            return True
        else:
            return False

    def __str__(self):
        return self.name+self.color
    

class Cards(object):
    """
    一副扑克牌类,54张牌,小王14-♠,大王15-♠
    """
    def __init__(self):
        #初始化扑克牌类型
        self.cards_type = ['1-♠-12', '1-♥-12','1-♣-12','1-♦-12',
                           '2-♠-13', '2-♥-13','2-♣-13','2-♦-13',
                           '3-♠-1', '3-♥-1','3-♣-1','3-♦-1',
                           '4-♠-2', '4-♥-2','4-♣-2','4-♦-2',
                           '5-♠-3', '5-♥-3','5-♣-3','5-♦-3',
                           '6-♠-4', '6-♥-4','6-♣-4','6-♦-4',
                           '7-♠-5', '7-♥-5','7-♣-5','7-♦-5',
                           '8-♠-6', '8-♥-6','8-♣-6','8-♦-6',
                           '9-♠-7', '9-♥-7','9-♣-7','9-♦-7',
                           '10-♠-8', '10-♥-8','10-♣-8','10-♦-8',
                           '11-♠-9', '11-♥-9','11-♣-9','11-♦-9',
                           '12-♠-10', '12-♥-10','12-♣-10','12-♦-10',
                           '13-♠-11', '13-♥-11','13-♣-11','13-♦-11',
                           '14-♠-14', '15-♠-15']
        #初始化扑克牌类                  
        self.cards = self.get_cards()

    #初始化扑克牌类
    def get_cards(self):
        cards = []
        for card_type in self.cards_type:
            cards.append(Card(card_type))
        #打乱顺序
        #np.random.shuffle(cards)
        return cards


############################################
#              出牌相关类                   #
############################################
class Feiji(object):
    """
    飞机类
    """
    def __init__(self, start, end, with_type, with_cards):
        self.start = start
        self.end = end
        self.with_type = with_type    # 飞机带牌数量，可选0 1 2
        self.with_cards = with_cards    # 飞机带的牌

class Qiegeji(object):
    """
    切割机类
    """
    def __init__(self, start, end):
        self.start = start
        self.end = end

class Moves(object):
    """
    出牌类,单,对,三,三带一,三带二,顺子,炸弹
    """ 
    def __init__(self):
        #出牌信息
        self.dan = []
        self.dui = []
        self.san = []
        self.san_dai_yi = []
        self.san_dai_er = []
        self.bomb = []
        self.shunzi = []
        self.sidai2 = []
        self.sidai2dui = []
        self.feiji = []
        self.qiegeji = []
        
        #牌数量信息
        self.card_num_info = {}
        #牌顺序信息,计算顺子
        self.card_order_info = []
        #王牌信息
        self.king = []
        
        #下次出牌
        self.next_moves = []
        #下次出牌类型
        self.next_moves_type = []
        
    #获取全部出牌列表
    def get_total_moves(self, cards_left):
        
        #统计牌数量/顺序/王牌信息
        for i in cards_left:
            #王牌信息
            if i.rank in [14,15]:
                self.king.append(i)
            #数量
            tmp = self.card_num_info.get(i.name, [])
            if len(tmp) == 0:
                self.card_num_info[i.name] = [i]
            else:
                self.card_num_info[i.name].append(i)
            #顺序
            if i.rank in [13,14,15]: #不统计2,小王,大王
                continue
            elif len(self.card_order_info) == 0:
                self.card_order_info.append(i)
            elif i.rank != self.card_order_info[-1].rank:
                self.card_order_info.append(i)
        
        #王炸
        if len(self.king) == 2:
            self.bomb.append(self.king)
            
        #出单,出对,出三,炸弹(考虑拆开)
        for _, v in self.card_num_info.items():
            if len(v) == 1:
                self.dan.append(v)
        for _, v in self.card_num_info.items():
            if len(v) == 2:
                self.dui.append(v)
                self.dan.append(v[:1])
        for _, v in self.card_num_info.items():
            if len(v) == 3:
                self.san.append(v)
                self.dui.append(v[:2])
                self.dan.append(v[:1])
        for _, v in self.card_num_info.items():
            if len(v) == 4:
                self.bomb.append(v)
                self.san.append(v[:3])
                self.dui.append(v[:2])
                self.dan.append(v[:1])
                
        #三带一,三带二
        for san in self.san:
            #if self.dan[0][0].name != san[0].name:
            #    self.san_dai_yi.append(san+self.dan[0])
            #if self.dui[0][0].name != san[0].name:
            #    self.san_dai_er.append(san+self.dui[0])
            for dan in self.dan:
                #防止重复
                if dan[0].name != san[0].name:
                    self.san_dai_yi.append(san+dan)
            for dui in self.dui:
                #防止重复
                if dui[0].name != san[0].name:
                    self.san_dai_er.append(san+dui)  
                    
        #获取最长顺子
        max_len = []
        for i in self.card_order_info:
            if i == self.card_order_info[0]:
                max_len.append(i)
            elif max_len[-1].rank == i.rank - 1:
                max_len.append(i)
            else:
                if len(max_len) >= 5:
                   self.shunzi.append(max_len) 
                max_len = [i]
        #最后一轮
        if len(max_len) >= 5:
           self.shunzi.append(max_len)   
        #拆顺子 
        shunzi_sub = []             
        for i in self.shunzi:
            len_total = len(i)
            n = len_total - 5
            #遍历所有可能顺子长度
            while(n > 0):
                len_sub = len_total - n
                j = 0
                while(len_sub+j <= len(i)):
                    #遍历该长度所有组合
                    shunzi_sub.append(i[j:len_sub+j])
                    j = j + 1
                n = n - 1
        self.shunzi.extend(shunzi_sub)
                
    #获取下次出牌列表
    def get_next_moves(self, last_move_type, last_move)-> (List[str], List[List[Card]]): 
        #没有last,全加上,除了bomb最后加
        if last_move_type == "start":
            moves_types = ["dan", "dui", "san", "san_dai_yi", "san_dai_er", "shunzi"]
            i = 0
            for move_type in [self.dan, self.dui, self.san, self.san_dai_yi, 
                      self.san_dai_er, self.shunzi]:
                for move in move_type:
                    self.next_moves.append(move)
                    self.next_moves_type.append(moves_types[i])
                i = i + 1
        #出单
        elif last_move_type == "dan":
            for move in self.dan:
                #比last大
                if move[0].bigger_than(last_move[0]):
                    self.next_moves.append(move)  
                    self.next_moves_type.append("dan")
        #出对
        elif last_move_type == "dui":
            for move in self.dui:
                #比last大
                if move[0].bigger_than(last_move[0]):
                    self.next_moves.append(move) 
                    self.next_moves_type.append("dui")
        #出三个
        elif last_move_type == "san":
            for move in self.san:
                #比last大
                if move[0].bigger_than(last_move[0]):
                    self.next_moves.append(move) 
                    self.next_moves_type.append("san")
        #出三带一
        elif last_move_type == "san_dai_yi":
            for move in self.san_dai_yi:
                #比last大
                if move[0].bigger_than(last_move[0]):
                    self.next_moves.append(move)    
                    self.next_moves_type.append("san_dai_yi")
        #出三带二
        elif last_move_type == "san_dai_er":
            for move in self.san_dai_er:
                #比last大
                if move[0].bigger_than(last_move[0]):
                    self.next_moves.append(move)   
                    self.next_moves_type.append("san_dai_er")
        #出炸弹
        elif last_move_type == "bomb":
            for move in self.bomb:
                #比last大
                if move[0].bigger_than(last_move[0]):
                    self.next_moves.append(move) 
                    self.next_moves_type.append("bomb")
        #出顺子
        elif last_move_type == "shunzi":
            for move in self.shunzi:
                #相同长度
                if len(move) == len(last_move):
                    #比last大
                    if move[0].bigger_than(last_move[0]):
                        self.next_moves.append(move) 
                        self.next_moves_type.append("shunzi")
        else:
            print("last_move_type_wrong")
            
        #除了bomb,都可以出炸
        if last_move_type != "bomb":
            for move in self.bomb:
                self.next_moves.append(move) 
                self.next_moves_type.append("bomb")
                
        return self.next_moves_type, self.next_moves
    
    
    #展示
    def show(self, info):
        print(info)
        #card_show(self.dan, "dan", 2)
        #card_show(self.dui, "dui", 2)
        #card_show(self.san, "san", 2)
        #card_show(self.san_dai_yi, "san_dai_yi", 2)
        #card_show(self.san_dai_er, "san_dai_er", 2)
        #card_show(self.bomb, "bomb", 2)
        #card_show(self.shunzi, "shunzi", 2)
        #card_show(self.next_moves, "next_moves", 2)

############################################
#              玩家相关类                   #
############################################        
class Agent(object):
    """
    玩家类,所有模型都应继承此类并重写choose方法
    """
    def __init__(self, player_id):
        self.player_id = player_id  # 0代表地主，1代表地主下家，2代表地主上家
        self.cards_left = []
        self.cards_out = []

    def set_game(self, game):
        self.game = game

    # 模型选择如何出牌
    def choose(self, state: 'State') -> (str, List[Card]):
        return None, None
        
    # 选牌，返回下一步可能出的所有牌的类型 具体牌内容
    def get_moves(self, last_move_type, last_move):
        #所有出牌可选列表
        self.total_moves = Moves()
        #获取全部出牌列表
        self.total_moves.get_total_moves(self.cards_left)
        #获取下次出牌列表
        self.next_move_types, self.next_moves = self.total_moves.get_next_moves(last_move_type, last_move)        

        return self.next_move_types, self.next_moves
    
    # 进行一步之后的公共操作
    def __common_step(self, last_move_type, last_move):
        #移除出掉的牌
        if self.next_move_type in ["yaobuqi", "buyao"]:
            self.next_move = []
        for i in self.next_move:
            self.cards_left.remove(i) 

        #是否牌局结束
        end = False
        if len(self.cards_left) == 0:
            end = True

        #要不起&不要
        if self.next_move_type in ["yaobuqi","buyao"]:
            yaobuqi = True
            next_move_type = last_move_type
            next_move = last_move
        else:
            yaobuqi = False
            next_move_type = self.next_move_type
            next_move = self.next_move
        return next_move_type, next_move, end, yaobuqi

    # 出牌
    def step(self, state):
        #在next_moves中选择出牌方法
        self.next_move_type, self.next_move = self.choose(state)
        self.game.cards_out.append( (self.player_id, self.next_move_type, self.next_move) )
        return self.__common_step(state.last_move_type, state.last_move)


############################################
#              状态相关类                   #
############################################        
class State(object):
    def __init__(self, player_id: int, cards_left: List[Card], 
                       cards_out: List[Tuple[int, str, List[Card]]], 
                       last_move_type: str, 
                       last_move: List[Card]):
        self.player_id = player_id
        self.cards_left = cards_left
        self.cards_out = cards_out
        self.last_move_type = last_move_type
        self.last_move = last_move
