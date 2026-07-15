"""
#!/usr/bin/env python
# -*- coding:utf-8 -*-
@Project : MASA-QMIX
@File : environment.py
@Author : Chenasuny
@Time : 2024/3/1 22:13
"""
import csv
import math
import random
import gym
# from gym import spaces
# from gym.utils import seeding
import utils.CTM.ctm_start as cs
from MARL.common.arguments import get_common_args
# from utils.CTM.ctm_simulate import baseGraph, nodeInfo
from utils.CTM.utils import calculate_fire_risk, load_all_firebytime, random_people
# import pandas as pd

#环境类
class DynamicSignalEnv(gym.Env):
    environment_name = "dynamic signal"
    def __init__(self,args):
        self.args = args
        self.baseGraph = None
        # 下面两个参数还不知道什么意思
        self.reward_threshold = -1000
        self.actions_record_for_agant = []
        self.fire_current = []
        # 一个全局状态，一个观测
        self.state4marl = None  # 维护全局state的变量
        self.obs4marl = None # 维护每个智能体局部观测的变量
        self.reset(load_all_firebytime())
        self.static_field = None
        self.fire_levels = None
        self.congestion_levels = None


    def reset(self,fire_info=None):
        """
        初始化环境（重置环境）
        得到初始化后的全局观测和局部观测
        """
        self.rannum = random.random()
        self.step_count = 0
        self.done = False
        # 加载火的信息
        self.all_firebytime = fire_info
        self.baseGraph = cs.init(self.all_firebytime)
        self.episode_time_slice = []
        self.action_space = [0,1,2,3]
        #获取全局状态
        self.state4marl= self._get_state_list()
        #获取局部状态
        self.obs4marl = self._get_obs_list()


    def _get_state_list(self):
        """
        全局的状态
        """
        all_nodesinfo = self.baseGraph.nodesinfo
        all_people_number = 0
        #静态场
        static_field = []
        fire_levels = []
        congestion_levels = []
        for i in all_nodesinfo.values():
            N = round(i.current_number)
            all_people_number = all_people_number + N
            static_field.append(N*(round(i.energy_domine/17,3)))
            _,fire_level = calculate_fire_risk(i.current_fireinfo[0],i.current_fireinfo[1],i.current_fireinfo[2],i.current_fireinfo[3])
            fire_levels.append(fire_level*N)  #用了火灾大小

            congestion_level = 0
            if i.current_density > 2.5:
                congestion_level = round((i.current_density -2.5 )/(6-2.5),3)
            congestion_levels.append(N*congestion_level)
        self.current_people_number = all_people_number
        self.static_field = static_field[:-3]
        self.fire_levels = fire_levels[:-3]
        self.congestion_levels = congestion_levels[:-3]
        cc = [x + y + z for x, y, z in zip(static_field ,congestion_levels, fire_levels)]

        return [all_people_number] + cc #

    def _get_obs_list(self):
        """
        每个智能体的观测
        """
        all_nodesinfo = self.baseGraph.nodesinfo

        fire_levels = []
        congestion_levels = []

        for i in range(len(all_nodesinfo)):
            node = all_nodesinfo[i + 1]
            _, fire_level = calculate_fire_risk(node.current_fireinfo[0], node.current_fireinfo[1],
                                                node.current_fireinfo[2], node.current_fireinfo[3])
            fire_levels.append(fire_level)
            congestion_levels.append(round(node.current_number))
        # 拥挤和火的信息
        return [congestion_levels[:-3]  + fire_levels[:-3] for _ in self.baseGraph.agent_cell_ids] #


    def step(self, actions):
        self.step_count += 1
        self.baseGraph.from_actions_get_groupId_submatrix(actions)  # 得到所有的方向
        self.baseGraph = cs.start_Sub_CTM(self.baseGraph,self.step_count)
        if self.current_people_number == 0  and  self.step_count > 20:
            self.done = True
        self.state4marl= self._get_state_list()
        self.obs4marl = self._get_obs_list()
        static_sum = sum(self.static_field)
        fire_sum = sum(self.fire_levels)
        cong_sum = sum(self.congestion_levels)
        # 奖励归一化开关 (成果4)：off 时与原始代码逐字节一致；on 时用归一化三分量奖励。
        # info 始终返回原始三分量，保证不同 reward 版本的指标可比。
        if getattr(self.args, 'reward_norm', False):
            p_ref = getattr(self.args, 'p_ref', 217.0)
            E = static_sum / p_ref   # 静态场（疏散效率）
            F = fire_sum / p_ref     # 火灾暴露
            C = cong_sum / p_ref     # 拥堵
            reward = -(0.4 * E + 0.4 * F + 0.2 * C)
        else:
            reward = - static_sum - fire_sum - cong_sum
        return  reward, self.done, [static_sum, fire_sum, cong_sum]
    def save_env_info(self, actions):
        self.actions_record_for_agant.append(actions)


    def get_state(self):
        assert self.state4marl is not None
        return self.state4marl

    def get_obs(self):
        # assert self.obs4marl is not None and self.obs4marl != []
        return  [self.get_obs_agent(i) for i in range(len(self.baseGraph.agent_cell_ids))]


    def get_obs_agent(self, agent_id):
        return self.obs4marl[agent_id]

    def get_env_info(self):
        return {
            "n_actions": len(self.action_space ),
            "n_agents": len(self.baseGraph.agent_cell_ids),
            "state_shape": len(self.get_state()),
            "obs_shape": len(self.get_obs()[0]),
            "episode_limit": 150
        }

    def get_avail_agent_actions(self, agent_id):
        available_action = [0 for x in range(len(self.action_space))]
        for i in self.baseGraph.signal_available_direction[agent_id]:
            available_action[i-1] = 1
        return available_action

if __name__ == '__main__':
    pass
