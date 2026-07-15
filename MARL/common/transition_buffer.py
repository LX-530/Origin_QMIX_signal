"""
Transition replay buffer for the CDQN-style autoregressive DQN (成果3).

与 QMIX 的 episode 级 ReplayBuffer 不同，这里存的是单步 transition：
    s        全局状态      [state_shape]
    a        联合动作      [n_agents]        每个 agent 的动作编号 (0..n_actions-1)
    r        标量奖励
    s_next   下一状态      [state_shape]
    done     是否真正终止  (到达 episode_limit 的时间截断不算 done)

动作掩码 avail 在环境里时不变、跨 episode 不变，因而不逐 transition 存储：
由 CascadeRunner 缓存一份 [n_agents, n_actions]，
且 avail_next == avail，训练时直接复用。
"""
import numpy as np


class TransitionBuffer:
    def __init__(self, capacity, state_shape, n_agents):
        self.capacity = int(capacity)
        self.state_shape = state_shape
        self.n_agents = n_agents

        self.s = np.zeros([self.capacity, state_shape], dtype=np.float32)
        self.a = np.zeros([self.capacity, n_agents], dtype=np.int64)
        self.r = np.zeros([self.capacity, 1], dtype=np.float32)
        self.s_next = np.zeros([self.capacity, state_shape], dtype=np.float32)
        self.done = np.zeros([self.capacity, 1], dtype=np.float32)

        self._idx = 0
        self._size = 0

    def __len__(self):
        return self._size

    @property
    def size(self):
        return self._size

    def push(self, s, a, r, s_next, done):
        i = self._idx
        self.s[i] = s
        self.a[i] = a
        self.r[i] = r
        self.s_next[i] = s_next
        self.done[i] = 1.0 if done else 0.0
        self._idx = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size):
        n = min(batch_size, self._size)
        idx = np.random.randint(0, self._size, size=n)
        return {
            's': self.s[idx],
            'a': self.a[idx],
            'r': self.r[idx],
            's_next': self.s_next[idx],
            'done': self.done[idx],
        }

    def state_dict(self):
        """Return the populated replay data for a resumable checkpoint."""
        n = self._size
        return {
            'capacity': self.capacity,
            'state_shape': self.state_shape,
            'n_agents': self.n_agents,
            'idx': self._idx,
            'size': n,
            's': self.s[:n].copy(),
            'a': self.a[:n].copy(),
            'r': self.r[:n].copy(),
            's_next': self.s_next[:n].copy(),
            'done': self.done[:n].copy(),
        }

    def load_state_dict(self, state):
        expected = (self.capacity, self.state_shape, self.n_agents)
        actual = (int(state['capacity']), int(state['state_shape']), int(state['n_agents']))
        if actual != expected:
            raise ValueError('Replay buffer shape mismatch: checkpoint {} != current {}'.format(actual, expected))

        n = int(state['size'])
        if not 0 <= n <= self.capacity:
            raise ValueError('Invalid replay buffer size: {}'.format(n))
        expected_shapes = {
            's': (n, self.state_shape),
            'a': (n, self.n_agents),
            'r': (n, 1),
            's_next': (n, self.state_shape),
            'done': (n, 1),
        }
        for key, shape in expected_shapes.items():
            if tuple(state[key].shape) != shape:
                raise ValueError('Replay field {} has shape {}, expected {}'.format(
                    key, tuple(state[key].shape), shape))

        self.s[:n] = state['s']
        self.a[:n] = state['a']
        self.r[:n] = state['r']
        self.s_next[:n] = state['s_next']
        self.done[:n] = state['done']
        self._idx = int(state['idx'])
        self._size = n
