import json
import os
import random
import re

import numpy as np
import torch

from env.environment import DynamicSignalEnv
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
from MARL.runner import Runner
from MARL.common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, \
    get_reinforce_args, \
    get_commnet_args, get_g2anet_args, get_cascade_args
import multiprocessing

import matplotlib.pyplot as plt

def marl_agent_wrapper():
    args = get_common_args()

    if args.alg.find('coma') > -1:  # 判断模型的参数
        args = get_coma_args(args)
    elif args.alg.find('central_v') > -1:
        args = get_centralv_args(args)
    elif args.alg.find('reinforce') > -1:
        args = get_reinforce_args(args)
    elif args.alg.find('cascade') > -1:  # CDQN-style autoregressive DQN
        args = get_cascade_args(args)
    else:
        args = get_mixer_args(args)
    if args.alg.find('commnet') > -1:
        args = get_commnet_args(args)
    if args.alg.find('g2anet') > -1:
        args = get_g2anet_args(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 加载调度环境
    env = DynamicSignalEnv(args)
    # env.reset()
    env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"] #4
    args.n_agents = env_info["n_agents"] #19
    args.state_shape = env_info["state_shape"] #143
    args.obs_shape = env_info["obs_shape"] #25
    args.episode_limit = env_info["episode_limit"]
    print("\n是否加载模型（测试必须）：", args.load_model, "是否打印中间变量：", args.havelook, "是否训练：",args.learn)

    # 用来保存plt和pkl
    save_path = args.result_dir + '/' + args.alg
    if args.resume_checkpoint:
        if args.alg.find('cascade') == -1:
            raise ValueError('--resume_checkpoint is only supported by cascade_dqn')
        if args.load_model:
            raise ValueError('--resume_checkpoint and --load_model cannot be used together')
        checkpoint_path = os.path.abspath(os.path.expanduser(args.resume_checkpoint))
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError('Resume checkpoint not found: {}'.format(checkpoint_path))
        model_dir = os.path.dirname(checkpoint_path)
        if os.path.basename(model_dir) != 'model':
            raise ValueError('Checkpoint must be inside <run_dir>/model: {}'.format(checkpoint_path))
        train_save_path = os.path.dirname(model_dir)
        args.resume_checkpoint = checkpoint_path
    else:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path_dir = get_next_folder(save_path, args.map + "_")
        train_save_path = save_path + "/" + save_path_dir
    train_save_path_model = train_save_path + '/model'
    train_save_path_actions = train_save_path + '/historydata/actions_result'
    train_save_path_data = train_save_path + '/historydata/train_process'
    if not os.path.exists(train_save_path):
        os.makedirs(train_save_path)
    if not os.path.exists(train_save_path_model):
        os.makedirs(train_save_path_model)
    if not os.path.exists(train_save_path_data):
        os.makedirs(train_save_path_data)
    if not os.path.exists(train_save_path_actions):
        os.makedirs(train_save_path_actions)
    args.save_path = train_save_path
    parameters = vars(args)
    # 指定要保存的文件名
    parameter_name = 'resume_parameters.json' if args.resume_checkpoint else 'training_parameters.json'
    filename = train_save_path + '/' + parameter_name
    # 写入JSON文件
    with open(filename, 'w') as file:
        json.dump(parameters, file, indent=4)

    # CDQN 走独立的单进程 transition-based 训练循环，完全绕开 QMIX 的 Runner/Agents/episode buffer
    if args.alg.find('cascade') > -1:
        from MARL.cascade_runner import CascadeRunner
        runner = CascadeRunner(env, args)
        try:
            runner.run(0)
        finally:
            runner.close()
        return

    runner = Runner(env, args)

    if args.learn:
        runner.run(0)  # 原来跑多种算法，run传入的是算法的id

        # 假设这些是你的损失值，存储为字符串（如你所提供）
        with open(train_save_path+"/historydata/loss.txt", 'r') as file:
            lines = file.readlines()
        # 提取数值
        loss_values = [float(loss.split('(')[1].split(',')[0]) for loss in lines]
        # 绘制图形
        plt.figure(figsize=(10, 5))
        plt.plot(loss_values, marker='o', linestyle='-', color='b', markerfacecolor='black', markersize=1)
        plt.title('Loss per Time Step')
        plt.xlabel('Time Step')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(train_save_path+"/historydata/loss.png", dpi=300)
    else:
        _, reward = runner.evaluate(10101)
        print('The ave_reward of {} is  {}'.format(args.alg, reward))

def get_next_folder(base_path, folder_prefix):
    # 列出base_path下的所有项
    dirs = os.listdir(base_path)
    # 使用正则表达式找到符合特定格式的文件夹
    pattern = re.compile(f"^{folder_prefix}(\\d+)$")
    max_num = 0
    for dir in dirs:
        match = pattern.match(dir)
        if match:
            # 提取数字并更新最大值
            num = int(match.group(1))
            if num > max_num:
                max_num = num
    # 返回下一个编号的文件夹名
    return f"{folder_prefix}{max_num + 1}"
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    marl_agent_wrapper()
