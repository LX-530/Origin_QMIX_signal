# 成果7：最终汇报 PPT 结构 + 可替换结果页

> 说明：本文件是 PPT 的骨架与素材清单。带 🔁 的是「可替换结果页」——
> 先放占位，服务器跑完全量后用 `analyze_compare.py` 的输出替换即可。

---

## 运行命令（服务器全量；本地小样把步数改小即可）

前置：在 repo root 运行；确保 `torch/gym/shapely` 可用。

```bash
# 1) 原 QMIX（未修正基线，raw reward）
python main.py --alg qmix --preset qmix_orig

# 2) 修正 QMIX（公平基线，归一化 reward）
python main.py --alg qmix --preset qmix_fixed --reward_norm

# 3) CDQN（归一化 reward，200k env steps）
python main.py --alg cascade_dqn --reward_norm

# CDQN 学习率候选（成果5）：改 arguments.get_cascade_args 里的 args.lr，或加 CLI 覆盖
#   5e-5 / 1e-4 / 3e-4

# 4) 生成对比图 + 汇总表（把下面路径换成实际 run 目录）
python analyze_compare.py --runs \
    orig=./result/qmix/dynamicsignal_1 \
    fixed=./result/qmix/dynamicsignal_2 \
    cdqn=./result/cascade_dqn/dynamicsignal_1 \
    --out ./result/compare
```

本地小样冒烟（几分钟，验证「可运行」）：

```bash
python main.py --alg cascade_dqn --reward_norm \
    --cli_total_env_steps 3000 --cli_warmup 500 --cli_anneal_env_steps 2000
python main.py --alg qmix --preset qmix_fixed --reward_norm --cli_n_epoch 5
python main.py --alg qmix --preset qmix_orig --cli_n_epoch 5
```

产出的指标 CSV：
- CDQN：`result/cascade_dqn/<run>/historydata/cascade_{train,eval}_metrics.csv`
- QMIX：`result/qmix/<run>/historydata/qmix_eval_metrics.csv`
- 统一列：`completion, evac_steps, reward_sum, fire_cum, cong_cum, static_cum`

---

## PPT 10 页结构

### 1. 研究问题
24 智能体下 QMIX 联合动作空间爆炸、训练不收敛，希望用级联 DQN 把联合动作搜索复杂度降为线性。

### 2. 论文启发（`1812.10613`）
Cascading DQN 通过 `Q_1 → Q_k` 依次选择组合动作，把组合动作搜索转为线性级联搜索。
本项目动作可重复（不同信号可选同一方向编号），故命名为 **CDQN-style autoregressive DQN**。

### 3. 当前代码审计（详见 `AUDIT.md`）
- 仓库实际为 **18 智能体**（非 24）。
- 训练不收敛两大高危原因：**epsilon 退火过快**、**训练更新不足**。
- 维度表：18 / 4 / 72 / 136 / 150。

### 4. 方法设计
- CDQN-style autoregressive DQN；保留信号 action mask（非法动作会使 CTM 崩溃，掩码是硬约束）。
- Double DQN + target network + Huber(SmoothL1) loss。
- 联合 Q（VDN 式级联求和）：`Q(s,a) = Σ_k q_k(s, a_<k)[a_k]`。

### 5. 网络结构图
```
state(72) ─► [Linear 72→128→128 +ReLU+LayerNorm] ─► h0 (GRU 初始 hidden, 128)
                                                        │
 for k = 1..18:                                         ▼
   concat( agent_emb[k] (18→16), action_emb[a_{k-1}] (5→16) ) ─► GRUCell(32→128) ─► h_k
                                                        │
                                                        ▼
                                          q_head: 128→64→4  ─► mask ─► argmax ─► a_k
```
文件：`MARL/network/cascade_net.py`、`MARL/policy/cascade_dqn.py`。

### 6. 级联动作选择流程图
```
a_1 = argmax_legal Q_1(s)
a_2 = argmax_legal Q_2(s, a_1)
...
a_k = argmax_legal Q_k(s, a_1,...,a_{k-1})
```
探索：每步以 ε 概率在**合法动作**内随机（保证不触发 CTM KeyError）。

### 7. 复杂度对比
| | 联合动作评估 |
|---|---|
| QMIX（枚举联合动作） | 最多 `4^18`（24 智能体则 `4^24`） |
| CDQN（级联评估） | 最多 `18 × 4`（24 智能体则 `24 × 4`） |

### 8. 奖励与状态
归一化三分量奖励（成果4，`--reward_norm`）：
```
E = Σ N_i·energy_i/17 / P_ref     # 疏散效率(静态场)
F = Σ N_i·fire_risk_i / P_ref     # 火灾暴露
C = Σ N_i·congestion_i / P_ref    # 拥堵
r = -(0.4·E + 0.4·F + 0.2·C)      # 安全优先备选: -(0.3E+0.55F+0.15C)
```
第一版用归一化 72 维 state；138 维 state 作为第二阶段开关（本次不做）。
为什么不加终点奖励/复杂 shaping：遵守奥卡姆剃刀，先归一化再谈复杂奖励。

### 9. 🔁 18 智能体实验结果
- 🔁 三算法同预算对比图 ← `result/compare/compare_curves.png`
- 🔁 CDQN 三个学习率对比图（5e-5 / 1e-4 / 3e-4）
- 🔁 关键指标表 ← `result/compare/summary_table.md`

| 算法 | 完成率 | 平均疏散步数 | 火灾暴露 | 拥堵 | 训练稳定性 |
|---|---:|---:|---:|---:|---|
| 原 QMIX | 待填 | 待填 | 待填 | 待填 | 待填 |
| 修正 QMIX | 待填 | 待填 | 待填 | 待填 | 待填 |
| CDQN | 待填 | 待填 | 待填 | 待填 | 待填 |

### 10. 结论 + 下一步
- 结论：CDQN 是否比修正 QMIX 更稳定；当前最大瓶颈来自算法 / 奖励 / 状态 / 还是 24 智能体定义缺失。
- 下一步：补齐 24 智能体定义、扩展场景种子、做 5 seed 稳定性、据结果决定是否替换 QMIX。

---

## 可替换素材清单（占位 → 替换来源）
| 素材 | 替换来源 |
|---|---|
| 当前代码审计表 | `AUDIT.md` 第 2/3 节 |
| 级联动作选择流程图 | 本文件第 6 页 |
| CDQN 网络结构图 | 本文件第 5 页 |
| reward 分量公式 | 本文件第 8 页 |
| QMIX vs CDQN 复杂度对比 | 本文件第 7 页 |
| 18 智能体训练曲线 | 🔁 `result/compare/compare_curves.png` |
| 18 智能体指标对比表 | 🔁 `result/compare/summary_table.md` |
| 24 智能体曲线或缺失说明 | 本次为「缺失说明」（`AUDIT.md` P7） |
| 最终结论页 | 本文件第 10 页 |

## 本次交付未覆盖（理想完成版）
- 24 智能体 reset/step/短训练（成果6，需先补齐 `agent_cell_ids` / 动作映射 / action mask / 影响 cell）。
- 138 维 state 切换；复杂 reward shaping；多进程 cascade 采集加速。
- 5 seed 稳定性评估（基于策略/初始化 seed）。
