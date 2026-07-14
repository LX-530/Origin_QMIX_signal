"""
#!/usr/bin/env python
# -*- coding:utf-8 -*-
@Project : MASA-QMIX
@File : ctm_simulate.py
@Author : Chenasuny
@Time : 2024/3/1 22:11
"""
import math
from shapely.geometry import Polygon
import  copy

free_velocity = 1.2  # 自由流速度
l = 1.7
max_density = 6  # 最大密度值 单位：人/米方
cri_density = 2.5  # 临界密度值 单位：人/米方
duration = l / free_velocity  # 时间步长，单位：秒


class nodeInfo():
    def __init__(self,cell_id,exit_inf, cell_inf, num_inf,neighbor_ids, fire_info,energy_domine):
        self.cell_id = cell_id
        self.exit = exit_inf
        self.cell = cell_inf
        self.neighbor_ids = neighbor_ids
        self.initial_number = num_inf  # 疏散开始时元胞内行人数
        if not self.exit:
            self.area = float(Polygon(cell_inf).area) # / 100  # 面积
        else:
            self.area = float("inf")
        self.initial_density = float(num_inf[0]) / float(self.area)  # 疏散开始时元胞内行人密度
        #需要每一步之后进行更新
        self.current_density = self.initial_density
        self.current_number =  self.initial_number[0]
        self.current_fireinfo = fire_info
        self.energy_domine =energy_domine
    #根据新密度更新数据
    def current_info(self,current_density,current_fire):
        self.current_density = current_density
        if self.exit:
            self.current_number = 0
        else:
            self.current_number =  self.current_density * self.area
        self.current_fireinfo = current_fire


class baseGraph():
    def __init__(self, exit_inf, cell_inf, init_num_inf, adjacency_mat,sigal_effect_cells,fixed_direction_cell,fire_info,agent_cell_ids,energy_domine):
        #实例化每一个node
        self.nodesinfo = {}
        for i in range(len(exit_inf)):
            nodeinfo = nodeInfo(i,exit_inf[i], cell_inf[i], init_num_inf[i],self.get_neighbors(adjacency_mat,i),self.get_fireinfo(fire_info,i),energy_domine[i])
            self.nodesinfo[i+1] = nodeinfo
        self.exit_info = exit_inf
        self.cell_inf = cell_inf
        self.init_num_inf = init_num_inf
        self.current_num = [0 for _ in init_num_inf]
        self.exs_list = []
        self.am = adjacency_mat  # 无向图的邻接矩阵
        self.fixed_direction_cell = fixed_direction_cell
        self.sigal_effect_cells = sigal_effect_cells
        self.signal_available_direction = self.get_signal_available_direction()  # 每个signal可以选择的方向
        self.all_direction_cells = []
        self.get_group_adjacency_matrix = []
        self.cell_number = len(self.nodesinfo)
        self.groups_ids = []
        self.sub_adj_matrixs = []
        self.groups_directions =[]
        self.fire_info_all = fire_info
        self.fire_info_current = fire_info[0]
        self.agent_cell_ids = agent_cell_ids
        self.energy_domine = energy_domine


    def get_signal_available_direction(self):
        signal_available_directions = []
        for key in self.sigal_effect_cells:
            signal_available_directions.append(list(key.keys()))
        return signal_available_directions


    def get_neighbors(self, adj_matrix, node):
        """
        返回指定节点的所有相邻节点及其接触长度
        :param adj_matrix: 邻接矩阵，二维列表
        :param node: 指定的节点编号（从0开始）
        """
        neighbors = {1:[],2:[],3:[],4:[]}
        for i, contact_info in enumerate(adj_matrix[node]):
            contact_direction = contact_info[0]
            if contact_direction >= 0.1:
                neighbors[contact_direction] =  neighbors[contact_direction] + [(i+1, contact_info[1])]
        return neighbors

    def get_fireinfo(self,fire_info,i):
        """
        初始化的火灾信息
        """
        return  [fire_info[0][0][i],fire_info[0][1][i],fire_info[0][2][i],fire_info[0][3][i]]


    def from_actions_get_groupId_submatrix(self, actions):
        """
        从动作中获得方向对
        """
        self.all_direction_cells = []
        directions = []
        for i, action in enumerate(actions):
            action_dict = self.sigal_effect_cells[i]
            direction = action_dict[int(action) + 1]
            directions.extend(direction)
        all_direction_cells = directions + self.fixed_direction_cell

        remove_repeat_direction_cells = copy.copy(all_direction_cells)
        for i_direction in all_direction_cells:
            x1,x2 = i_direction
            for j_direction in all_direction_cells:
                y1, y2 = j_direction
                if x1==y2 and x2==y1:
                    if i_direction in  remove_repeat_direction_cells:
                        remove_repeat_direction_cells.remove(i_direction)
                        remove_repeat_direction_cells.remove(j_direction)

        self.all_direction_cells = remove_repeat_direction_cells
        # 下面得到子图的邻接矩阵
        self.directions2matrix() #
        self.form_adj_matrix_get_subgroup_ids() #
        self.subgroup_ids2sub_adj_matrix()



    def directions2matrix(self):

        adjacency_matrix = [
            [0] * self.cell_number for _ in range(self.cell_number)
        ]
        # 根据提供的边在邻接矩阵中标记相应的位置
        for edge in self.all_direction_cells:
            start_node, end_node = edge
            adjacency_matrix[start_node-1][end_node-1] = 1
        self.get_group_adjacency_matrix = adjacency_matrix


    def form_adj_matrix_get_subgroup_ids(self):
        """
        从邻接矩阵中得到子图IDs
        """
        groups = {}
        conbining_group = {}

        for i in range(len(self.get_group_adjacency_matrix)):
            node_neighbor = self.get_group_adjacency_matrix[i]
            indices_list = [i for i, x in enumerate(node_neighbor) if x == 1]
            indices = set(indices_list)
            for key, value in groups.items():
                intersection = value & set(indices)
                if i in value or intersection:
                    value_new = value | set(indices) | {i, }
                    groups[key] = value_new
                    indices = {}
            if indices != {}:
                value_new = indices | {i, }
                groups[len(groups)] = value_new
        for key, value in conbining_group.items():
            combine_item = list(value)[0]
            combine_dict = {}
            value_in_combining_group = groups[combine_item]
            combine_dict[combine_item] = value_in_combining_group
            for i in value:
                value_in_combining_group = groups.pop(i)
                combine_dict[combine_item] = combine_dict[combine_item] | value_in_combining_group
            groups.update(combine_dict)

        ids_set = {}
        for value in groups.values():
            ids_set = set(ids_set) | value
        if len(ids_set) < len(self.get_group_adjacency_matrix):
            for i in range(len(self.get_group_adjacency_matrix)):
                intersection = ids_set & set(i)
                if intersection == {}:
                    groups[len(groups) + len(self.get_group_adjacency_matrix)] = {i, }

        groups_ids = [list(xx) for xx in list(groups.values())]


        removal_repeat_groups = []
        list_group = []
        for i , igroup in enumerate(groups_ids):
            for j , jgroup in enumerate(groups_ids):
                intset = set(igroup) & set(jgroup)
                if i != j and len(intset)>0:
                    removal_repeat_groups.append([i,j])
                    list_group.append(i)
                    list_group.append(j)

        finianl_groups_ids = []


        for ii in removal_repeat_groups:
            merge = set(groups_ids[ii[0]]) | set(groups_ids[ii[1]])
            finianl_groups_ids.append(list(merge))
            bb=[ii[1],ii[0]]
            removal_repeat_groups.remove(bb)

        for jj,group  in enumerate( groups_ids):
            if jj not in list_group:
                finianl_groups_ids.append(group)

        self.groups_ids = finianl_groups_ids
        self.from_subgroup2directions()

    def from_subgroup2directions(self):
        """
        从group的Id中得到有效的方向对
        """
        all_directions_cells = self.all_direction_cells
        groups_directions = []
        for group in self.groups_ids:
            group_directions = []
            for i in group:
                direction = [xy for xy in all_directions_cells if i+1 in xy]
                group_directions.extend(direction)
            unique_list = []
            for item in group_directions:
                if item not in unique_list:
                    unique_list.append(item)
            groups_directions.append(unique_list)
        self.groups_directions = groups_directions


    def subgroup_ids2sub_adj_matrix(self):

        sub_am =[]
        groups_direction = self.groups_directions
        for j, group_direction in enumerate(groups_direction):
            group_id = self.groups_ids[j]
            adj_matrix = [[(0, 0.0, 0.0)] * len(group_id) for _ in range(len(group_id))]
            for x,y in group_direction:
                adj_matrix[group_id.index(x-1)][group_id.index(y-1)] = self.am[x-1][y-1]
            sub_am.append(adj_matrix)
        self.sub_adj_matrixs = sub_am


    def update_node_num_info(self,finished_subgroups, step):
        fire_info_all = self.fire_info_all[step]
        for sub_group in finished_subgroups:
            nodes = sub_group.nodes
            for node in nodes:
                fire_info = [fire_info_all[0][node.real_ID-1],fire_info_all[1][node.real_ID-1],fire_info_all[2][node.real_ID-1],fire_info_all[3][node.real_ID-1]]
                self.nodesinfo[node.real_ID].current_info(node.density,fire_info)
        for id, nd in self.nodesinfo.items():
            self.current_num[id-1] = nd.current_number
        self.fire_info_current = fire_info_all
