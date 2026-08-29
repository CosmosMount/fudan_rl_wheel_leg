# 训练加速实验

本记录使用 RTX 4060 Laptop GPU（55 W）和 8192 个平地环境。短训练均从同一配置、
seed 1 开始；绝对时间会受功耗和后台负载影响，结论以配对结果为主。

## 已采用

- MuJoCo-Warp 接触容量：`nconmax` 从 200 调整为 64，`njmax` 保持 200。48 步配对
  rollout 从 97.99 ms/控制步降到 90.87 ms，吞吐从 83,600 提高到 90,155
  env-step/s（+7.8%）。平地 8192 环境与跳跃 4096 环境压力测试的单环境接触峰值
  分别为 16/64 和 15/64；约束峰值分别为 86/200 和 82/200。
- PPO 复用历史编码：encoder 在 PPO 阶段冻结，因此同一 batch 的 actor 和 critic
  共享一次 latent 与原算法等价。学习阶段短测约从 0.53 s/iteration 降到
  0.49 s/iteration；采样仍是主要瓶颈。

接触容量的 200 步、8192 环境确定性动作对照显示，`nconmax=64` 与重复运行默认
`nconmax=200` 的差异处于 MuJoCo-Warp 跨进程非确定性噪声内；优化配置的平均观测
差异反而小于两次默认配置之间的差异。容量峰值监控是更直接的溢出安全检查。

## 未作为默认值

| 实验 | 短测结果 | 决策 |
| --- | --- | --- |
| `njmax: 200 -> 128` | 96.53 vs 97.99 ms/控制步，差异接近噪声 | 保留 200 |
| PPO epoch `5 -> 3` | 学习约 0.53 -> 0.32 s/iteration | 会改变更新比率，保留 5 |
| 轻量时序 encoder/actor/critic | 学习约 0.53 -> 0.35 s/iteration | 改变模型容量，保留为架构实验 |

同时使用 `nconmax=64, njmax=128`、共享 latent 和 3 个 PPO epoch 的 50-iteration
实验耗时约 4:16，基线约 4:54（快 12.9%），但它同时改变了更新比率，不能把早期
回报差异归因于某一项。保守的 5-epoch 组合延长到 iteration 148 后快约 8.5%，
统一评估中偏航跟踪弱于基线，因此最终拆分参数，只采用具有明确容量余量和配对吞吐
收益的 `nconmax=64`，并保留 5 个 epoch 与原网络容量。

实验分支：

- `codex/fast-sim`：接触/约束容量实验。
- `codex/fused-ppo`：共享冻结 latent。
- `codex/low-update-ppo`：3-epoch PPO。
- `codex/light-temporal`：轻量时序架构。

运行采样基准：

```bash
python scripts/benchmark_rollout.py --mode plane --num-envs 8192
python scripts/benchmark_rollout.py --mode jump --num-envs 4096 --steps 500
```
