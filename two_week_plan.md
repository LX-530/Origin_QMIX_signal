# CDQN 改造内容总结

## 目标

把当前工程中与级联 DQN 相关的内容整理清楚：新增了哪些文件、改了哪些入口、训练流程怎么走、目前能输出什么结果。

当前 CDQN 指的是：

```text
Cascading DQN / CDQN-style autoregressive DQN
```

核心思想：每个环境步内，18 个信号智能体按固定顺序依次选动作。

```text
a1 = argmax Q1(s)
a2 = argmax Q2(s, a1)
...
ak = argmax Qk(s, a1, ..., a{k-1})
```

这样把联合动作搜索从近似 `4^18` 降为 `18 x 4`。

## 当前环境确认

当前仓库实际是 **18 个智能体**，不是 24 个。

确认位置：

- 智能体编号：`utils/CTM/ctm_start/__init__.py`

```python
agent_cell_ids = [32, 43, 46, 50, 51, 57, 60, 62, 18, 13, 11, 10, 8, 5, 25, 17, 30, 66]
```

- 环境读取：`env/environment.py`

```python
"n_agents": len(self.baseGraph.agent_cell_ids)
```

训练参数中确认：

```text
n_agents = 18
n_actions = 4
state_shape = 72
obs_shape = 136
episode_limit = 150
```

## 新增文件

### 1. `MARL/network/cascade_net.py`

作用：定义 CDQN 的级联 Q 网络。

网络结构：

```text
state encoder: state_shape -> 128 -> 128
agent embedding: n_agents -> 16
action embedding: n_actions + START -> 16
cascade core: GRUCell
q head: 128 -> 64 -> n_actions
```

关键点：

- 用 `GRUCell` 逐个智能体展开。
- 第 k 个智能体的输入包含：
  - 全局状态编码；
  - 当前智能体编号；
  - 上一个智能体已选动作。
- 输出每个智能体的 4 个动作 Q 值。
- 使用 action mask 屏蔽非法动作。
- 联合 Q 值采用逐智能体求和：

```text
joint_Q(s, a) = sum q_k(s, a_<k)[a_k]
```

### 2. `MARL/policy/cascade_dqn.py`

作用：定义 CDQN 策略、动作选择、Double DQN 学习目标和模型保存。

主要内容：

- 创建 `eval_net` 和 `target_net`。
- 使用 Adam 优化器。
- `select_actions()`：按级联顺序选择 18 个智能体动作。
- `learn()`：从 transition batch 中训练网络。
- `save_model()`：保存模型参数。

为什么用 Double DQN：

```text
online net 负责选动作
target net 负责估值
```

目标值：

```text
a_next = argmax Q_online(s_next)
y = r + gamma * (1 - done) * Q_target(s_next, a_next)
loss = SmoothL1(Q_online(s, a), y)
```

作用：减少普通 DQN 的 Q 值高估，提升多智能体级联训练稳定性。

### 3. `MARL/common/transition_buffer.py`

作用：新增 transition 级经验池。

与原 episode 级 buffer 不同，这里保存单步 transition：

```text
s        当前全局状态
a        18 个智能体联合动作
r        标量奖励
s_next   下一全局状态
done     是否真正完成疏散
```

注意：

- 到达 `episode_limit=150` 只是时间截断，不等于真正完成。
- action mask 不进 buffer，因为当前环境中 mask 固定，由 runner 缓存。

### 4. `MARL/cascade_runner.py`

作用：CDQN 的独立训练循环。

它走独立训练流程，不使用原有的：

- `Runner`
- `Agents`
- episode replay buffer
- 多进程 `Pool`

训练流程：

```text
reset env
读取 action mask
选择级联动作
env.step(actions)
写入 transition buffer
warmup 后每 train_every 步训练一次
每 20 episode 做一次 greedy eval
保存模型和指标
```

输出文件：

```text
result/cascade_dqn/.../historydata/cascade_train_metrics.csv
result/cascade_dqn/.../historydata/cascade_eval_metrics.csv
result/cascade_dqn/.../historydata/loss.txt
result/cascade_dqn/.../model/*_cascade_net_params.pkl
```

记录指标：

```text
episode
env_step
completion
evac_steps
reward_sum
fire_cum
cong_cum
static_cum
epsilon
loss
illegal
```

## 修改文件

### 1. `main.py`

新增 CDQN 入口。

当运行：

```bash
python main.py --alg cascade_dqn
```

程序会进入：

```python
from MARL.cascade_runner import CascadeRunner
runner = CascadeRunner(env, args)
runner.run(0)
```

即 CDQN 走独立训练流程，不再走原有 Runner。

### 2. `MARL/common/arguments.py`

新增 CDQN 参数入口：

```python
--alg cascade_dqn
--cli_total_env_steps
--cli_warmup
--cli_anneal_env_steps
```

新增 `get_cascade_args(args)`。

当前主要超参数：

| 参数 | 当前值 |
|---|---:|
| learning rate | `1e-4` |
| gamma | `0.99` |
| batch size | `256` |
| replay buffer | `100000` |
| warmup | `5000` |
| train_every | `4` |
| target_update | `1000` |
| grad clip | `10` |
| loss | `SmoothL1` |

### 3. `env/environment.py`

新增或保留了归一化奖励开关：

```bash
--reward_norm
```

归一化奖励：

```text
E = static_sum / P_ref
F = fire_sum / P_ref
C = congestion_sum / P_ref

r = -(0.4 * E + 0.4 * F + 0.2 * C)
```

同时 `info` 仍保留原始三分量，方便对比：

```text
static_cum
fire_cum
cong_cum
```

### 4. `analyze_compare.py`

作用：读取评估 CSV，生成曲线图和汇总表。

本次主要用于 CDQN 500 轮结果整理：

```text
result/compare_500/compare_curves.png
result/compare_500/summary_table.md
result/compare_500/summary_interpretation.md
```

## 当前已跑结果

运行口径：

```bash
python main.py --alg cascade_dqn --reward_norm --cli_total_env_steps 75000 --cli_warmup 5000 --cli_anneal_env_steps 75000
```

含义：

```text
500 episode x 150 step = 75000 env steps
```

结果目录：

```text
result/cascade_dqn/dynamicsignal_2
```

关键结果：

| 指标 | 数值 |
|---|---:|
| eval rows | 25 |
| first 5 eval reward avg | `-34.8052` |
| last 10 eval reward avg | `-27.0067` |
| reward improvement | `22.41%` |
| final reward | `-27.6705` |
| final completion | `0` |
| final evac_steps | `150` |
| illegal actions | `0` |

结论：

- CDQN 能跑通。
- 奖励有改善。
- 没有非法动作。
- 但完成率仍为 0，说明还没学到完整疏散策略。

## 目前最重要的问题

1. 当前只有 18 个智能体定义，24 智能体还缺：
   - `agent_cell_ids`
   - 动作映射
   - action mask
   - 每个信号影响的 cell

2. 500 轮 CDQN 仍未完成疏散。

3. 当前结果只能说明 CDQN 初版可运行，不能说明已经解决收敛问题。

## 后续最小工作

1. 继续补跑 CDQN 多组设置：

```text
learning rate: 5e-5 / 1e-4 / 3e-4
reward weight: 保守版 / 安全优先版
seed: 至少 3 个
```

2. 固定记录：

```text
completion
evac_steps
reward_sum
fire_cum
cong_cum
illegal
```

3. 若 CDQN 仍为 0 完成率，优先改奖励：

```text
加入完成疏散终局奖励
加大 fire 权重
降低拥堵项对 early learning 的干扰
```

4. 若扩展到 24 智能体，先补环境定义，再训练。

## 一句话总结

本次 CDQN 改造新增了 `cascade_net.py`、`cascade_dqn.py`、`transition_buffer.py`、`cascade_runner.py`，并在 `main.py` 和 `arguments.py` 中接入 `--alg cascade_dqn`。当前版本已能在 18 智能体环境中完成 500 episode 训练，但还没有实现成功疏散。
