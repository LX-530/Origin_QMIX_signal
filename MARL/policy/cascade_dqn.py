"""
CDQN-style autoregressive DQN policy (成果3).

- 联合 Q（VDN 式级联求和）: joint_Q(s, a) = sum_k q_k(s, a_<k)[a_k]
- Double DQN 目标:
      a_next = CascadeArgmax(Q_online, s_next)      # online 选
      y      = r + gamma * (1 - done) * Q_target(s_next, a_next)   # target 评
      loss   = SmoothL1(Q_online(s, a), y)
- 全程 action mask（掩码时不变，由 runner 缓存后 set_avail_mask 传入）。

走独立训练循环 CascadeRunner，不使用 QMIX 的 episode buffer / RNN / 多进程。
"""
import os
import torch
import torch.nn.functional as F

from MARL.network.cascade_net import CascadeNet


class CascadeDQN:
    def __init__(self, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_shape = args.state_shape
        self.gamma = args.gamma
        self.target_update = args.target_update
        self.grad_norm_clip = args.grad_norm_clip

        self.eval_net = CascadeNet(args)
        self.target_net = CascadeNet(args)

        if args.cuda and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is not available')
        self.device = torch.device('cuda:0' if args.cuda else 'cpu')
        self.eval_net.to(self.device)
        self.target_net.to(self.device)

        self.model_dir = args.load_model_dir + '/' + args.alg + '/' + args.map
        if getattr(args, 'load_model', False):
            path = self.model_dir + '/cascade_net_params.pkl'
            if os.path.exists(path):
                try:
                    model_state = torch.load(path, map_location=self.device, weights_only=True)
                except TypeError:
                    model_state = torch.load(path, map_location=self.device)
                self.eval_net.load_state_dict(model_state)
                print('Successfully load the model:', path)
            else:
                raise Exception("No model!")

        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=args.lr)

        # 静态动作掩码 [n_agents, n_actions]，由 runner 在 reset 后调用 set_avail_mask 设置
        self.avail_mask = None
        self.grad_steps = 0

    def set_avail_mask(self, avail_mask):
        """avail_mask: numpy/list [n_agents, n_actions] (0/1)。掩码时不变，只需设一次。"""
        self.avail_mask = torch.tensor(avail_mask, dtype=torch.float32, device=self.device)

    def select_actions(self, state, epsilon):
        """
        state: numpy [state_shape]；返回 actions numpy [n_agents], joint_q float。
        """
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        avail = self.avail_mask.unsqueeze(0)                # [1, n_agents, n_actions]
        actions, joint_q = self.eval_net.select(s, avail, epsilon)
        return actions.squeeze(0).cpu().numpy(), float(joint_q.item())

    def learn(self, batch):
        """
        batch: dict of numpy arrays  s[B,state], a[B,n_agents], r[B,1], s_next[B,state], done[B,1]
        返回标量 loss。
        """
        s = torch.tensor(batch['s'], dtype=torch.float32, device=self.device)
        a = torch.tensor(batch['a'], dtype=torch.long, device=self.device)
        r = torch.tensor(batch['r'], dtype=torch.float32, device=self.device).squeeze(1)
        s_next = torch.tensor(batch['s_next'], dtype=torch.float32, device=self.device)
        done = torch.tensor(batch['done'], dtype=torch.float32, device=self.device).squeeze(1)
        B = s.shape[0]
        avail = self.avail_mask.unsqueeze(0).expand(B, -1, -1)   # [B, n_agents, n_actions]

        # 预测：online 对实际执行动作的联合 Q（可反传）
        q_eval = self.eval_net.score(s, a)                       # [B]

        # 目标：Double DQN —— online 选 a_next，target 评估
        with torch.no_grad():
            a_next, _ = self.eval_net.select(s_next, avail, epsilon=0.0)
            q_target_next = self.target_net.score(s_next, a_next)   # [B]
            y = r + self.gamma * (1.0 - done) * q_target_next

        loss = F.smooth_l1_loss(q_eval, y)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), self.grad_norm_clip)
        self.optimizer.step()

        self.grad_steps += 1
        if self.grad_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        return float(loss.item())

    def save_model(self, tag):
        save_dir = self.args.save_path + '/model'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        path = save_dir + '/' + str(tag) + '_cascade_net_params.pkl'
        torch.save(self.eval_net.state_dict(), path)
        return path

    def training_state_dict(self):
        return {
            'eval_net': self.eval_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'grad_steps': self.grad_steps,
        }

    def load_training_state_dict(self, state):
        self.eval_net.load_state_dict(state['eval_net'])
        self.target_net.load_state_dict(state['target_net'])
        self.optimizer.load_state_dict(state['optimizer'])
        for optimizer_state in self.optimizer.state.values():
            for key, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state[key] = value.to(self.device)
        self.grad_steps = int(state['grad_steps'])
