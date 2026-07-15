"""
CDQN-style autoregressive DQN network (成果3).

依次为所有信号选择动作：
    a_1 = argmax Q_1(s)
    a_2 = argmax Q_2(s, a_1)
    ...
    a_k = argmax Q_k(s, a_1, ..., a_{k-1})

结构（见 two_week_plan.md 成果3 / plan）：
    state encoder : state_dim -> 128 -> 128        (输出作 GRU 初始 hidden)
    agent embedding : n_agents -> 16               (按 cascade 位置 k 索引)
    action embedding : (n_actions+1) -> 16         (前一个已选动作；index n_actions = START)
    cascade core : GRUCell(agent_emb+action_emb, hidden=128)
    q head : 128 -> 64 -> n_actions

联合 Q 采用 VDN 式级联求和：joint_Q(s, a) = sum_k q_k(s, a_<k)[a_k]。
把联合动作评估从指数复杂度 (n_actions^n_agents) 降为线性 (n_agents * n_actions)。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


NEG_INF = -1e9  # 用有限大负数代替 -inf，避免 argmax/gather 出现 NaN


class CascadeNet(nn.Module):
    def __init__(self, args):
        super(CascadeNet, self).__init__()
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_shape = args.state_shape
        self.hidden = args.state_hidden
        self.start_token = self.n_actions  # 前置动作的 START 标记

        # state encoder -> GRU 初始 hidden
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_shape, self.hidden),
            nn.ReLU(),
            nn.LayerNorm(self.hidden),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.LayerNorm(self.hidden),
        )
        self.agent_emb = nn.Embedding(self.n_agents, args.agent_emb_dim)
        self.action_emb = nn.Embedding(self.n_actions + 1, args.action_emb_dim)
        self.gru = nn.GRUCell(args.agent_emb_dim + args.action_emb_dim, self.hidden)
        self.q_head = nn.Sequential(
            nn.Linear(self.hidden, args.q_hidden),
            nn.ReLU(),
            nn.Linear(args.q_hidden, self.n_actions),
        )

    def _device(self):
        return next(self.parameters()).device

    def _run(self, state, avail=None, epsilon=0.0, teacher_actions=None):
        """
        统一的自回归展开。
        state           : [B, state_shape]
        avail           : [B, n_agents, n_actions] (0/1)，select 时必需；score 时可为 None
        epsilon         : 探索率（仅 select 模式使用）
        teacher_actions : [B, n_agents] long，给定则为 score 模式（teacher forcing）
        返回 actions [B, n_agents] long, joint_q [B]
        """
        device = self._device()
        B = state.shape[0]
        h = self.state_encoder(state)                       # [B, hidden]
        prev_a = torch.full((B,), self.start_token, dtype=torch.long, device=device)
        joint_q = torch.zeros(B, device=device)
        actions = torch.zeros(B, self.n_agents, dtype=torch.long, device=device)

        for k in range(self.n_agents):
            agent_idx = torch.full((B,), k, dtype=torch.long, device=device)
            inp = torch.cat([self.agent_emb(agent_idx), self.action_emb(prev_a)], dim=1)
            h = self.gru(inp, h)                            # [B, hidden]
            q_k = self.q_head(h)                            # [B, n_actions]

            if teacher_actions is not None:
                a_k = teacher_actions[:, k]                 # 给定动作（保证合法）
            else:
                avail_k = avail[:, k, :]                    # [B, n_actions]
                q_masked = q_k.masked_fill(avail_k == 0, NEG_INF)
                greedy_a = q_masked.argmax(dim=1)           # 只在合法动作里贪心
                if epsilon > 0.0:
                    probs = avail_k / avail_k.sum(dim=1, keepdim=True)
                    rand_a = torch.multinomial(probs, 1).squeeze(1)   # 只从合法动作里随机
                    explore = torch.rand(B, device=device) < epsilon
                    a_k = torch.where(explore, rand_a, greedy_a)
                else:
                    a_k = greedy_a

            # 累加选中动作的 Q（a_k 恒合法，取未 mask 的原始 q 值）
            joint_q = joint_q + q_k.gather(1, a_k.unsqueeze(1)).squeeze(1)
            actions[:, k] = a_k
            prev_a = a_k

        return actions, joint_q

    @torch.no_grad()
    def select(self, state, avail, epsilon=0.0):
        """行为/贪心选动作。返回 actions [B, n_agents], joint_q [B]。"""
        return self._run(state, avail=avail, epsilon=epsilon, teacher_actions=None)

    def score(self, state, actions):
        """teacher forcing：给定 actions，返回可反传的 joint_q [B]。"""
        _, joint_q = self._run(state, avail=None, epsilon=0.0, teacher_actions=actions)
        return joint_q
