# MADE-QMIX

Multi-agent dynamic evacuation


# Implementation
```python
python main.py
```

Hyper-paras can be found in './MARL/common/arguments.py'

## 算法与实验（最小成功版本）

**必须在 repo root 运行**（数据用相对路径）。

```bash
# 原 QMIX（未修正基线）
python main.py --alg qmix --preset qmix_orig
# 修正 QMIX（公平基线：epsilon 1.0->0.05 over ~200k、train_steps=20、归一化奖励）
python main.py --alg qmix --preset qmix_fixed --reward_norm
# CDQN-style autoregressive DQN（成果3）
python main.py --alg cascade_dqn --reward_norm

# 环境冒烟
python tools/smoke_test.py
# 对比曲线 + 汇总表
python analyze_compare.py --runs orig=<run_dir> fixed=<run_dir> cdqn=<run_dir> --out ./result/compare
```

本地小样冒烟可加预算覆盖：`--cli_total_env_steps`、`--cli_warmup`、`--cli_anneal_env_steps`（CDQN）、`--cli_n_epoch`（QMIX）。

### W&B 实时监控（CDQN）

服务器只需登录一次，API Key 不要写入代码：

```bash
pip install "wandb>=0.22.3,<0.23"
wandb login
python main.py --alg cascade_dqn --reward_norm --cuda True \
  --cli_total_env_steps 375000 --cli_anneal_env_steps 200000 \
  --wandb --wandb_project QMIX-signal --wandb_run_name cdqn-long-2500
```

W&B 页面实时显示训练/评估的 reward、completion、evac_steps、fire、congestion、epsilon、loss 和 illegal；断网测试可加 `--wandb_mode offline`。

### CDQN 断点续训

训练默认每 5000 个梯度步覆盖保存 `<run_dir>/model/latest_checkpoint.pt`，并在正常结束时再保存一次。它包含 online/target 网络、优化器、梯度步数、环境步数、episode 编号、回放缓冲区、随机数状态和 W&B run ID。

```bash
python main.py --alg cascade_dqn --reward_norm --cuda True \
  --cli_total_env_steps 600000 \
  --resume_checkpoint ./result/cascade_dqn/dynamicsignal_1/model/latest_checkpoint.pt \
  --wandb --wandb_project QMIX-signal
```

`--cli_total_env_steps` 是续训后的总目标步数，必须大于 checkpoint 中的 `env_step`。续训沿用原结果目录并追加 CSV；在线 W&B 会继续原 run。只加载自己生成、可信的 checkpoint。

新增/改动文件：`MARL/network/cascade_net.py`、`MARL/policy/cascade_dqn.py`、
`MARL/common/transition_buffer.py`、`MARL/cascade_runner.py`、`analyze_compare.py`、
`tools/smoke_test.py`；`MARL/common/arguments.py`、`main.py`、`env/environment.py`、
`MARL/runner.py`、`MARL/common/rollout.py`（后两者仅新增评估指标记录）。
