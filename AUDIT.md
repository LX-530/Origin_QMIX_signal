# 成果1：环境与问题审计

> 目的：把当前代码和训练问题讲清楚，避免直接把「不收敛」归因到 QMIX 或 24 智能体数量。
> 结论先行：**当前仓库是 18 智能体，不是 24 智能体**；训练不收敛的两大高危原因是
> **epsilon 退火过快** 与 **训练更新次数不足**，与算法本身无必然关系。

## 1. 环境检查记录

| 项 | 结果 |
|---|---|
| Python | 3.12.9（服务器建议对齐；本地已验证可运行） |
| 依赖 | `torch 2.5.1`、`numpy 2.3.2`、`gym 0.26.2`、`shapely 2.1.2` 均可用 |
| numpy 2.x 兼容 | ✅ 安全，全仓库无 `np.float/np.int/np.bool` 等弃用用法 |
| gym | 仅作 `gym.Env` 基类使用（`spaces/seeding` 已注释），0.26 可用 |
| 运行目录 | **必须在 repo root 运行**：`load_all_firebytime()` 以相对路径读取 `./fire_info/building_devc_fire_in_cell_27.csv` |
| 火灾数据 | `fire_info/building_devc_fire_in_cell_27.csv`，含 **690** 个时间步（远多于 episode_limit=150） |

## 2. 环境维度表（真实参数，冒烟测试实测）

| 参数 | 值 | 来源 |
|---|---:|---|
| n_agents | **18** | `agent_cell_ids` 18 个（`ctm_start/__init__.py:61`）；`sigal_effect_cells` 中 agent10 被注释 |
| n_actions | **4** | `action_space=[0,1,2,3]`（`environment.py:53`） |
| state_shape | **72** | `[all_people_number] + cc`，cc 为 71 个 cell（`environment.py:87`）→ 1+71 |
| obs_shape | **136** | `congestion[:-3] + fire[:-3]` = 68+68（`environment.py:105`） |
| episode_limit | **150** | `get_env_info`（`environment.py:140`） |

> ⚠️ `main.py:37-40` 的内联注释写的是 `n_agents=19 / state=143 / obs=25`，**是过时错误注释**，与实际代码不符。已在本次交付中通过 env_info 自动读取，不再依赖这些注释。

## 3. 当前代码问题表

| # | 问题 | 位置 | 影响 | 本次处理 |
|---|---|---|---|---|
| P1 | **epsilon 退火过快**：`anneal_steps=1` → `anneal_epsilon=0.8`，1 个 epoch 内 epsilon 从 1.0 直接掉到 min 0.2 | `arguments.py`(原) | 探索几乎立刻停止，早期就锁死次优策略 | 新增 `--preset qmix_fixed`：1.0→0.05 over ~200k env steps（0.02/epoch） |
| P2 | **训练更新次数不足**：每轮采集 `28×150≈4200` env steps，但 `train_steps=1` 只更新 1 次 | `arguments.py`(原) | 样本严重欠拟合，学习极慢 | `qmix_fixed` 将 `train_steps` 提到 20 |
| P3 | min_epsilon 偏高（0.2） | `arguments.py`(原) | 收敛后仍有 20% 随机 | `qmix_fixed` 降到 0.05 |
| P4 | 硬编码 Linux 路径 `/home/dell/桌面/...` | `arguments.py:30-31`(原) | 换机即崩，不可移植 | 改为相对路径 `./result`、`./MARL/model` |
| P5 | 优先经验回放实为死代码：`terminated` 已 padding 到 150，`shape[0]>=145` 恒真 → 优先级永不生效 | `replay_buffer.py:40` | 「完成 episode 加权」从未触发 | 记录在案（不影响本次最小版本） |
| P6 | epsilon 的 step/episode 尺度退火在多进程 Pool 子进程里被丢弃，只有 per-epoch 递减生效 | `rollout.py` / `runner.py:79` | 只能用 epoch 尺度退火 | `qmix_fixed` 采用 epoch 尺度并据此计算 0.02/epoch |
| P7 | **24 智能体定义缺失**：仓库只有 18 个 `agent_cell_ids` / `sigal_effect_cells` | `ctm_start/__init__.py` | 无法直接跑 24 智能体 | 本次不做（成果6），仅在此标注 |

## 4. 随机合法动作冒烟测试结果

`python tools/smoke_test.py`（repo root）：

```
env_info: {'n_actions': 4, 'n_agents': 18, 'state_shape': 72, 'obs_shape': 136, 'episode_limit': 150}
reset ~4.6 ms | step ~3.6 ms
150-step episode ~0.54 s | 200k env steps ~12 min (steps only)
rollout ok: total_reward=-6914.65, steps_no_legal_action=0, done=False
SMOKE TEST PASSED
```

关键结论：
- 18 个信号的 150 步随机**合法**动作全程无崩溃，**非法动作数=0**。
- **动作掩码是强约束**：`ctm_simulate/__init__.py:110` 做 `action_dict[int(action)+1]`，**任何非法动作直接 KeyError 崩溃**，因此贪心与探索都必须只在合法动作内选择。
- 动作掩码 `signal_available_direction` 在环境初始化时计算一次，**时不变、跨 episode 不变**；每个 agent 至少 1 个合法动作、没有 agent 拥有全部 4 个动作 → 训练时可缓存 `[18,4]` 掩码，且 `avail_next == avail`。
- 原始奖励量级：单 episode 累计约 **−6900**（O(10⁴)），由静态场主导，拥堵在随机策略下常为 0。**该量级过大不利于 DQN 拟合** → 已提供 `--reward_norm` 归一化开关（÷P_ref，见成果4），用于修正 QMIX 与 CDQN 的公平对比。

## 5. 环境的一个重要特性：近似确定性

- `random_people` 内部 `random.seed(10)`（`utils/__init__.py:91`）→ 初始人流布局**每个 episode 完全相同**。
- 火灾读固定 CSV。
- 因此在给定动作序列下环境**近似确定**，episode 之间只由探索不同。
- **含义**：后续「多 seed 稳定性」应基于策略/初始化 seed，而不是环境随机性；也解释了为何「场景太固定 → 收敛可能是记忆化」是一个需要注意的风险。

## 6. 可放入 PPT 的三条审计结论

1. **当前代码是 18 智能体，不是 24 智能体**（附维度表）。
2. **训练不收敛的两个高危原因**：epsilon 退火过快（P1）、训练更新不足（P2）——都属训练配置问题，先修正再评价算法。
3. **环境可跑通、掩码是硬约束、奖励量级过大**——这决定了 CDQN 与修正 QMIX 都需要归一化奖励作公平对照。
