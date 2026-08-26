# WBR mjlab

WBR 平地速度与跳跃任务的紧凑 mjlab 迁移。两套任务共享机器人、环境状态、观测、动作和训练实现，只通过配置覆盖保留原来的任务差异。

## 安装与命令

项目固定使用 `mjlab==1.6.0`、`rsl-rl-lib==5.4.2`。本机直接使用已有的
micromamba `wbr` 环境，不创建项目内 `.venv`：

```bash
micromamba activate wbr
python -m pip install --no-deps --no-build-isolation -e .
train Mjlab-Velocity-Flat-WBR
train Mjlab-Jump-Flat-WBR
play Mjlab-Velocity-Flat-WBR
play Mjlab-Jump-Flat-WBR
export-wbr-policy \
  --task Mjlab-Velocity-Flat-WBR \
  --checkpoint logs/rsl_rl/wbr_plane/<RUN>/model_<ITER>.pt \
  --output policy.onnx
pytest
```

全新环境在 editable 安装前还需一次性安装轻量构建工具：
`python -m pip install hatchling editables`。`uv.lock` 保留用于依赖版本审计和复现，
但日常命令不通过 `uv run` 执行。

mjlab 通过项目的 `mjlab.tasks` entry point 自动发现两个任务，无需修改 mjlab 本身。

## 固定接口

- `policy`: 当前 25 维本体观测。
- `history`: 单一 25 维 term 的 5 帧历史，按帧排列为 125 维。
- `critic`: 141 维特权观测。
- 动作顺序: `ljoint1, ljoint4, lwheel, rjoint1, rjoint4, rwheel`。
- ONNX: 输入 `obs[B,25]`、`obs_history[B,125]`，输出 `actions[B,6]`。

仿真使用 1 ms Euler/Newton 步长、20 次迭代、`1e-9` 容差；每次 policy step 固定执行 10 个物理子步。腿部在每个子步执行位置 PD，轮子执行速度 PD，两个气弹簧 tendon actuator 的控制量保持为零并由 MJCF bias 产生被动力。

## 文件职责

- `assets/wbr.xml`: 无 mesh、无外部路径的自包含碰撞/简洁可视 MJCF。
- `src/wbr_mjlab/robot.py`: 模型入口和控制系统常量。
- `src/wbr_mjlab/task.py`: plane/jump 共享任务工厂。
- `src/wbr_mjlab/mdp.py`: 状态缓存、观测、混合动作、奖励、随机化和课程。
- `src/wbr_mjlab/rl.py`: 最小 Sequence-PPO、续训状态和 ONNX 导出。
- `tests/test_migration.py`: 模型、MDP、环境、学习与导出验收。
- `assets/policies/legacy_*.onnx`: 仅归档的旧模型，不用于新 WBR 部署或续训。

已在 RTX 3090 上完成 GPU 验收：plane 8192 env 和 jump 4096 env 均运行 10 iterations，无 NaN 或闭链发散；两套任务的 checkpoint 恢复、play 推理步进与 ONNX 导出也均通过。
