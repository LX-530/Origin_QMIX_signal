"""
#!/usr/bin/env python
# -*- coding:utf-8 -*-
@Project : MASA-QMIX
@File : utils.py
@Author : Chenasuny
@Time : 2024/4/2 18:16
"""
import copy
import csv
import math
import multiprocessing
import random
import warnings

# import threading
import numpy as np
from MARL.agent.agent import Agents
from MARL.common.rollout import RolloutWorker


free_velocity = 1.2  # 自由流速度
l = 1.7
max_density = 6  # 最大密度值 单位：人/米方
cri_density = 2.5  # 临界密度值 单位：人/米方
duration = l / free_velocity  # 时间步长，单位：秒



def ped_fd(density: float):  # 跟据线性的行人流基本图，输入密度，输出流率
    if density <= cri_density:
        return density * free_velocity
    elif density <= max_density:
        return (10 - density) / (10 - cri_density) * cri_density * free_velocity
    else:
        return (10 - max_density) / (10 - cri_density) * cri_density * free_velocity


def calculate_fire_risk(Rad, Tem, Tox, Vis):
    """
    火灾对人的风险计算
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        beta1, beta2, beta3, beta4 = 1, 1, 1, 1
        r_rad = 1/(1+ np.exp(beta1 * (2.5- Rad)))
        r_tem = 1/(1+ np.exp(beta2 * (100- Tem)))
        r_tox = 1/(1+ np.exp(beta3 * (2800 - Tox)))
        r_vis = 1/(1+ np.exp(beta4 * (Vis - 5)))
        risk = round((r_rad+r_tem+r_tox+r_vis)/4,5) # using average score
    return float(risk), float(risk)   #没有用火灾等级


def fire_effect_speed( Tem, Tox, Vis, interal_time, v_current):
    """
    火灾对元胞间流动速度的影响
    interal_time:时间间隔
    v_current: 不考虑的情况下的速度
    return : 返回火对速度的影响系数
    """
    if v_current<1.2:
        v_current=1.2
    v_max = 2.0

    e_vis, e_co, e_temp = 0,0,0
    if Vis>=7.5:
        e_vis = 1
    elif Vis<7.5 and Vis >=2.5:
        e_vis = 1.375-(2.8125/Vis)
    elif  Vis < 2.5:
        e_vis = 0.25

    if Tox<0.1:
        e_co = 1
    elif Tox >= 0.1 and Tox <= 0.25:
        e_co = 1- (0.2125+1.788*Tox)*Tox*interal_time
    elif  Tox >0.25:
        e_co = 0

    if Tem<=30:
        e_temp = 1
    elif Tem<=60 and Tem>30:
        e_temp = 1+ (v_max-v_current)*(((Tem-30)/(60-30))**2)/v_current
    elif Tem<=120 and Tem>60:
        e_temp = (v_max/v_current)*(1-(((Tem-60)/(120-60))**2))

    return e_vis*e_co*e_temp


def random_people(list,fire_cell_number):
    random.seed(10)
    random_num = [random.randint(6, 12) for _ in range(25)]
    random_distance = [[int(random.uniform(1, 8)) for _ in range(i)] for i in random_num ]
    random_distance_people = []
    for j in range(71):  #71是总的元胞数
        if j+1 in list:
            random_distance_people.append(random_distance.pop())
        else:
            random_distance_people.append(0)
    random_distance_people_step = []
    for index, num_distance in enumerate(random_distance_people): #按照时间步及是否后置来控制元胞每一步进入人的数量
        delay_if = False
        if num_distance ==0:
            random_distance_people_step.append([0])
        else:
            count_num = []
            for ii in range(6):
                count = sum(1 for x in num_distance if ii*1.7 <= x <= (ii+1)*1.7)
                count_num.append(count)
            random_distance_people_step.append(count_num)

        #根据CTM基本图流动计算每一个时间步的流入元胞的人数
    final_flow = []
    for i in random_distance_people_step:
        if i == [0] :
            final_flow.append(i)
        else:
            outcell =[]
            k =0
            while True:
                k = k + 1
                add = 0
                if k < len(i):
                    add = i[k]
                a = i[0]+ add
                outflow =round(1 * ped_fd(a/1)*duration)
                outflow = min(a,outflow)
                outcell.append(outflow)
                a = a - outflow
                i[0] = a
                if a == 0 and k >= len(i):
                    break
            final_flow.append(outcell)
    return final_flow

def load_all_firebytime():
    all_fireinfo = []
    with open('./fire_info/building_devc_fire_in_cell_27.csv', 'r') as file:
        # 创建 CSV 读取器对象
        reader = csv.reader(file)
        # 逐行读取数据
        for step, row in enumerate(reader):
            if step < 2:
                continue
            step_fire = []
            # 打印每一行数据
            row = [round(float(element),5) for element in row] # 将str转换为float
            # row = [0 for element in row] # 将str转换为float
            
            rad =  [round(element * 1000000) for element in row[214:285]]
            # rad =  [0 for element in row[214:285]]
            
            rad[-1] =0
            rad[-2] =0
            rad[-3] =0
            step_fire.append(rad)  # radiation
            temp = row[72:143]
            # temp =[20 for i in row[72:143]]
            
            temp[-1] =20
            temp[-2] =20
            temp[-3] =20
            step_fire.append(temp)  # temperature
            tox = row[143:214]
            tox[-1] =0
            tox[-2] =0
            tox[-3] =0
            step_fire.append(tox)  # toxic
            vis = row[1:72]
            # vis =[30 for i in row[1:72]]
            vis[-1] =30
            vis[-2] =30
            vis[-3] =30
            step_fire.append(vis)  # visibility
            all_fireinfo.append(step_fire)
    return all_fireinfo

#多线程的类
class multiTask:
    def __init__(self, n_episodes, env,args ,policy, fire):
        self.env = env
        self.args = copy.deepcopy(args)
        self.args.load_model = False

        self.n_episodes = n_episodes
        self.agents= Agents(self.args)
        self.agents.agents_copy(policy)
        self.fire = fire

    def generate_episode(self, params):
        episode_idx, epoch = params
        rolloutWorker = RolloutWorker(self.env, self.agents, self.args)
        # 生成单个episode
        episode, _, _, for_gantt_data = rolloutWorker.generate_episode(epoch=epoch, episode_num=episode_idx,
                                                                            fire_info=self.fire)
        del rolloutWorker
        return episode, sum(episode['r'][0])[0]

    def run_simulation(self, epoch):
        episodes = []
        r_s = []
        batch_size = 28
        params = [(episode_idx, epoch) for episode_idx in range(self.n_episodes)]
        for i in range(0, len(params), batch_size):
            with multiprocessing.Pool(batch_size) as pool:
                batch_results = pool.map(self.generate_episode, params[i:i+batch_size])
            for episode, r in batch_results:
                episodes.append(episode)
                r_s.append(r)
        return episodes, r_s


if __name__ == '__main__':
    random.seed(10)
    random_num = [random.randint(6, 12) for _ in range(25)]
    print(sum(random_num))






