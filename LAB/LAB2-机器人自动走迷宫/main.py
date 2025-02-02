# 导入相关包
import os
import random
import numpy as np
from Maze import Maze
from Runner import Runner
from QRobot import QRobot
from ReplayDataSet import ReplayDataSet
from torch_py.MinDQNRobot import MinDQNRobot as TorchRobot # PyTorch版本
from keras_py.MinDQNRobot import MinDQNRobot as KerasRobot # Keras版本
import matplotlib.pyplot as plt


import numpy as np

# 机器人移动方向
move_map = {
    'u': (-1, 0), # up
    'r': (0, +1), # right
    'd': (+1, 0), # down
    'l': (0, -1), # left
}


# 迷宫路径搜索树
class SearchTree(object):


    def __init__(self, loc=(), action='', parent=None):
        """
        初始化搜索树节点对象
        :param loc: 新节点的机器人所处位置
        :param action: 新节点的对应的移动方向
        :param parent: 新节点的父辈节点
        """

        self.loc = loc  # 当前节点位置
        self.to_this_action = action  # 到达当前节点的动作
        self.parent = parent  # 当前节点的父节点
        self.children = []  # 当前节点的子节点

    def add_child(self, child):
        """
        添加子节点
        :param child:待添加的子节点
        """
        self.children.append(child)

    def is_leaf(self):
        """
        判断当前节点是否是叶子节点
        """
        return len(self.children) == 0


def expand(maze, is_visit_m, node):
    """
    拓展叶子节点，即为当前的叶子节点添加执行合法动作后到达的子节点
    :param maze: 迷宫对象
    :param is_visit_m: 记录迷宫每个位置是否访问的矩阵
    :param node: 待拓展的叶子节点
    """
    can_move = maze.can_move_actions(node.loc)
    for a in can_move:
        new_loc = tuple(node.loc[i] + move_map[a][i] for i in range(2))
        if not is_visit_m[new_loc]:
            child = SearchTree(loc=new_loc, action=a, parent=node)
            node.add_child(child)


def back_propagation(node):
    """
    回溯并记录节点路径
    :param node: 待回溯节点
    :return: 回溯路径
    """
    path = []
    while node.parent is not None:
        path.insert(0, node.to_this_action)
        node = node.parent
    return path


def my_search(maze):
    """
    深度优先搜索算法
    :param maze: 迷宫对象
    :return :到达目标点的路径 如：["u","u","r",...]
    """

    path = []

    # -----------------请实现你的算法代码--------------------------------------
    stack = [] # 创建⼀个空的栈
    start = maze.sense_robot()
    root = SearchTree(loc=start)
    stack = [root] # 根节点压⼊栈中
    h, w, _ = maze.maze_data.shape
    is_visit_m = np.zeros((h, w), dtype=np.int) # 标记迷宫的各个位置是否被访问过
    
    while stack:
        current_node = stack.pop() #从栈中取出当前节点
        is_visit_m[current_node.loc] = 1 # 标记当前节点位置已访问
        # 到达⽬标点
        if current_node.loc == maze.destination:
            path = back_propagation(current_node)
            break
        # 拓展叶节点
        if current_node.is_leaf():
            expand(maze, is_visit_m, current_node)
        # 将⼦节点⼊栈（逆序）
        for child in reversed(current_node.children):
            stack.append(child)
    # -----------------------------------------------------------------------
    return path


import os
import random
import numpy as np
import torch
from QRobot import QRobot
from ReplayDataSet import ReplayDataSet
from torch_py.MinDQNRobot import MinDQNRobot as TorchRobot # PyTorch版本
import matplotlib.pyplot as plt

class Robot(TorchRobot):

    def __init__(self, maze):
        """
        初始化 Robot 类
        :param maze:迷宫对象
        """
        super(Robot, self).__init__(maze)
        maze.set_reward(reward={
            "hit_wall": 10.,
            "destination": -maze.maze_size ** 2 *10,
            "default": 1.,
        })
        self.maze = maze
        self.epsilon = 0
        """开启金手指，获取全图视野"""
        self.memory.build_full_view(maze=maze)
        self.train()
        

    def train(self):      
        # 训练，直到能走出这个迷宫
        while True:
            self._learn(batch=len(self.memory) )
            success = False
            self.reset()
            for _ in range(self.maze.maze_size ** 2 ):
                a, r = self.test_update()
                if r == self.maze.reward["destination"]:
                    return 

    def train_update(self):
        state = self.sense_state()
        action = self._choose_action(state)
        reward = self.maze.move_robot(action)
        
        return action, reward
    
    
    def test_update(self):
        # 获取当前状态
        state = np.array(self.sense_state(), dtype=np.int16)
        # 根据Q表选择最佳动作
        action = self._choose_best_action(state)
        # 执行动作并获得奖励
        reward = self.maze.move_robot(action)
        # 返回动作和奖励
        return action, reward
    
    def _choose_best_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        q_values = self.eval_model(state)  # 假设这个方法返回给定状态下所有动作的Q值
        self.eval_model.eval()
        with torch.no_grad():
            best_action_index = np.argmin(q_values)
        return self.valid_action[best_action_index]  # 假设valid_actions是所有可能动作的列表
