import argparse

"""
Here are the param for the training
"""


def get_common_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--difficulty', type=str, default='7', help='the difficulty of the game')
    parser.add_argument('--game_version', type=str, default='latest', help='the version of the game')
    parser.add_argument('--map', type=str, default='dynamicsignal', help='the map of the game')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--step_mul', type=int, default=8, help='how many steps to make an action')
    parser.add_argument('--replay_dir', type=str, default=r'', help='absolute path to save the replay')
    # 一共可以测试13种算法，但是智能体实际上只有8种
    # The alternative algorithms are vdn, coma, central_v, qmix, qtran_base,
    # qtran_alt, reinforce, coma+commnet, central_v+commnet, reinforce+commnet，
    # coma+g2anet, central_v+g2anet, reinforce+g2anet, maven
    parser.add_argument('--alg', type=str, default='qmix', help='the algorithm to train the agent (qmix / cascade_dqn)')
    # 修正版 QMIX 基线开关：qmix_orig 保持原配置（未修正基线），qmix_fixed 使用修正后的 epsilon 退火与更新次数
    parser.add_argument('--preset', type=str, default='qmix_orig', choices=['qmix_orig', 'qmix_fixed'],
                        help='QMIX config preset: qmix_orig (unmodified baseline) or qmix_fixed (corrected baseline)')
    # 奖励归一化开关：off 时奖励与原始代码逐字节一致；on 时使用 two_week_plan 成果4 的归一化三分量奖励
    parser.add_argument('--reward_norm', action='store_true',
                        help='normalize reward by P_ref: r = -(0.4*E + 0.4*F + 0.2*C)')
    parser.add_argument('--p_ref', type=float, default=217.0,
                        help='reference population for reward normalization (env is deterministic; ~total evacuees)')
    # 可选的 Weights & Biases 实时监控；不开启时不导入 wandb，不影响原训练。
    parser.add_argument('--wandb', action='store_true', help='enable Weights & Biases experiment tracking')
    parser.add_argument('--wandb_project', type=str, default='QMIX-signal', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='W&B user/team; default uses logged-in account')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='optional W&B run name')
    parser.add_argument('--wandb_mode', type=str, default='online',
                        choices=['online', 'offline', 'disabled'], help='W&B synchronization mode')
    # 训练预算的 CLI 覆盖（本地小样冒烟用小值，服务器全量用大值）。不指定则用各算法预设默认值。
    parser.add_argument('--cli_total_env_steps', type=int, default=None, help='override CDQN total_env_steps')
    parser.add_argument('--cli_warmup', type=int, default=None, help='override CDQN warmup transitions')
    parser.add_argument('--cli_anneal_env_steps', type=int, default=None, help='override CDQN epsilon anneal horizon')
    parser.add_argument('--cli_save_cycle', type=int, default=None,
                        help='override CDQN checkpoint interval in gradient steps')
    parser.add_argument('--cli_n_epoch', type=int, default=None, help='override QMIX n_epoch')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='resume CDQN from <run_dir>/model/latest_checkpoint.pt')
    parser.add_argument('--last_action', type=bool, default=True, help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', type=bool, default=True, help='whether to use one network for all agents')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--optimizer', type=str, default="RMS", help='optimizer')
    parser.add_argument('--evaluate_epoch', type=int, default=20, help='number of the epoch to evaluate the agent')

    # 以下几个参数比较常用,注意只有指定变量名的时候，就会变成true，没法b变false，最好直接在这里改参数吧
    # parser.add_argument('--model_dir', type=str, default='./result/', help='model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./result', help='model directory of the policy')
    parser.add_argument('--load_model_dir', type=str, default='./MARL/model', help='result directory of the policy')

    parser.add_argument('--load_model', type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--learn', type=bool, default=True, help='whether to train the model')
    parser.add_argument('--cuda', type=bool, default=False, help='whether to use the GPU')
    parser.add_argument('--havelook', type=bool, default=False, help='whether to have a look of the saruo')

    args = parser.parse_args()
    return args

def get_mixer_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.qmix_hidden_dim = 64
    args.two_hyper_layers =  True
    args.hyper_hidden_dim = 64
    args.qtran_hidden_dim = 64
    args.lr = 5e-4

    # epsilon greedy
    args.epsilon = 1
    args.min_epsilon = 0.2
    anneal_steps = 1 #3000
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'epoch'

    # the number of the epoch to train the agent
    args.n_epoch =  2300 #5000  #15000

    # the number of the episodes in one epoch
    args.n_episodes = 28

    # the number of the train steps in one epoch
    args.train_steps = 1

    # # how often to evaluate
    args.evaluate_cycle = 1 #10

    args.evaluate_epoch = 1

    # experience replay
    args.batch_size =  320 #320
    args.buffer_size = int(5e3)

    # how often to save the model
    args.save_cycle = 300 #300

    # how often to update the target_net
    args.target_update_cycle =100 # 100

    # QTRAN lambda
    args.lambda_opt = 1
    args.lambda_nopt = 1

    # prevent gradient explosion
    args.grad_norm_clip = 10

    # MAVEN
    args.noise_dim = 16
    args.lambda_mi = 0.001
    args.lambda_ql = 1
    args.entropy_coefficient = 0.001

    # ---- 修正版 QMIX 基线 (成果2) ----
    # qmix_orig: 保持上面的原始配置（未修正基线）
    # qmix_fixed: 修正 epsilon 退火过快 + 训练更新不足两大问题
    if getattr(args, 'preset', 'qmix_orig') == 'qmix_fixed':
        # epsilon 1.0 -> 0.05，在约 200000 env steps 内退火。
        # 只能用 epoch scale：step/episode 尺度的退火发生在 Pool 子进程里会被丢弃，
        # 只有 runner.run 里每个 epoch 一次的 self.args.epsilon -= anneal_epsilon 会生效。
        # steps/epoch = n_episodes * episode_limit = 28 * 150 = 4200
        # epochs_to_anneal = 200000 / 4200 ≈ 47.6  ->  anneal_epsilon = (1.0-0.05)/47.6 ≈ 0.02
        args.epsilon = 1.0
        args.min_epsilon = 0.05
        args.anneal_epsilon = 0.02
        args.epsilon_anneal_scale = 'epoch'
        # 每轮采集约 4200 env steps 却只训练 1 次 -> 提高更新次数
        args.train_steps = 20
    if getattr(args, 'cli_n_epoch', None) is not None:
        args.n_epoch = args.cli_n_epoch
    return args


def get_cascade_args(args):
    """
    CDQN-style autoregressive DQN 的超参数 (成果3)。
    该算法走独立的 CascadeRunner（transition-based, Double DQN），
    不使用 QMIX 的 episode ReplayBuffer / RNN / 多进程 Pool。
    """
    # network
    args.state_hidden = 128        # state encoder 隐层，也是 GRU 的 hidden
    args.agent_emb_dim = 16        # agent 位置嵌入
    args.action_emb_dim = 16       # 前一个已选动作嵌入
    args.q_hidden = 64             # Q head 隐层

    # optimization
    args.optimizer = 'Adam'
    args.lr = 1e-4
    args.gamma = 0.99
    args.batch_size = 256          # transitions
    args.buffer_size = int(1e5)    # replay capacity (transitions)
    args.warmup = 5000             # transitions before first gradient step
    args.train_every = 4           # env steps per gradient step
    args.target_update = 1000      # gradient steps between hard target syncs
    args.grad_norm_clip = 10

    # exploration: epsilon 1.0 -> 0.05 over anneal_env_steps（单进程，可按 env step 退火）
    args.epsilon = 1.0
    args.min_epsilon = 0.05
    args.anneal_env_steps = 200000

    # training budget（本地小样冒烟用小值，服务器全量用大值）
    args.total_env_steps = 200000
    args.episode_limit = 150       # 若 env_info 覆盖则以 env_info 为准

    # logging / checkpoint
    args.log_every_episodes = 1
    args.save_cycle = 5000         # gradient steps between full checkpoints

    # CLI 覆盖（本地小样冒烟）
    if getattr(args, 'cli_total_env_steps', None) is not None:
        args.total_env_steps = args.cli_total_env_steps
    if getattr(args, 'cli_warmup', None) is not None:
        args.warmup = args.cli_warmup
    if getattr(args, 'cli_anneal_env_steps', None) is not None:
        args.anneal_env_steps = args.cli_anneal_env_steps
    if getattr(args, 'cli_save_cycle', None) is not None:
        args.save_cycle = args.cli_save_cycle
    if args.total_env_steps <= 0 or args.warmup <= 0 or args.anneal_env_steps <= 0 or args.save_cycle <= 0:
        raise ValueError('CDQN step budgets and save_cycle must be positive')

    return args

# arguments of coma
def get_coma_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'epoch'

    # lambda of td-lambda return
    args.td_lambda = 0.8

    # the number of the epoch to train the agent
    args.n_epoch = 20000

    # the number of the episodes in one epoch
    args.n_episodes = 1

    # how often to evaluate
    args.evaluate_cycle = 100

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args



# arguments of central_v
def get_centralv_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.05
    args.epsilon_anneal_scale = 'epoch'

    # the number of the epoch to train the agent
    args.n_epoch = 20000

    # the number of the episodes in one epoch
    args.n_episodes = 1

    # how often to evaluate
    args.evaluate_cycle = 100

    # lambda of td-lambda return
    args.td_lambda = 0.8

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of central_v
def get_reinforce_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'epoch'

    # the number of the epoch to train the agent
    args.n_epoch = 20000

    # the number of the episodes in one epoch
    args.n_episodes = 1

    # how often to evaluate
    args.evaluate_cycle = 100

    # how often to save the model
    args.save_cycle = 5000

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of coma+commnet
def get_commnet_args(args):
    if args.map == '3m':
        args.k = 2
    else:
        args.k = 3
    return args


def get_g2anet_args(args):
    args.attention_dim = 32
    args.hard = True
    return args
