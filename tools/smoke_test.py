"""
环境冒烟测试 (成果1 交付物)。

在 repo root 下运行：
    python tools/smoke_test.py

验证：
- env_info 维度：n_agents=18, n_actions=4, state_shape=72, obs_shape=136, episode_limit=150
- reset + 150 步「随机合法动作」不崩溃（非法动作会触发 CTM KeyError）
- reward == -(static + fire + cong)（原始奖励；info 始终返回原始三分量）
- reset / step 计时，用于估算训练时长
"""
import os
import sys
import time
import types
import random

# 允许从 repo root 直接运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import DynamicSignalEnv
from utils.CTM.utils import load_all_firebytime


def main():
    fire = load_all_firebytime()
    print('fire timesteps:', len(fire))

    env = DynamicSignalEnv(types.SimpleNamespace(reward_norm=False))
    info = env.get_env_info()
    print('env_info:', info)
    assert info == {'n_actions': 4, 'n_agents': 18, 'state_shape': 72,
                    'obs_shape': 136, 'episode_limit': 150}, info
    n = info['n_agents']

    # 计时 reset
    t0 = time.perf_counter(); env.reset(fire); reset_ms = (time.perf_counter() - t0) * 1000

    masks = [env.get_avail_agent_actions(a) for a in range(n)]
    legal = [[i for i, m in enumerate(mk) if m == 1] for mk in masks]
    assert all(len(l) >= 1 for l in legal), 'some agent has no legal action!'
    assert all(len(l) < 4 for l in legal), 'some agent has all 4 actions (unexpected)'

    illegal = 0
    r_sum = 0.0
    done = False
    t0 = time.perf_counter()
    for t in range(info['episode_limit']):
        acts = []
        for a in range(n):
            if not legal[a]:
                illegal += 1
                acts.append(0)
            else:
                acts.append(random.choice(legal[a]))
        r, done, comp = env.step(acts)
        # 原始奖励恒等式
        assert abs(r - (-(comp[0] + comp[1] + comp[2]))) < 1e-6, (r, comp)
        r_sum += r
        if done:
            break
    step_ms = (time.perf_counter() - t0) / (t + 1) * 1000

    print('reset ~{:.1f} ms | step ~{:.2f} ms'.format(reset_ms, step_ms))
    print('150-step episode ~{:.2f} s | 200k env steps ~{:.1f} min (steps only)'.format(
        (reset_ms + step_ms * 150) / 1000, step_ms * 200000 / 1000 / 60))
    print('rollout ok: total_reward={:.2f}, steps_no_legal_action={}, done={}'.format(r_sum, illegal, done))
    assert illegal == 0
    print('SMOKE TEST PASSED')


if __name__ == '__main__':
    main()
