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

## 原生 MuJoCo / ONNX 键盘推理

已提供独立的原生 MuJoCo 步进程序，可加载从 `.pt` 导出的 ONNX，通过键盘控制。
本机 `wbr` 环境已安装 ONNX Runtime，已有导出文件可直接运行：

```bash
python -m wbr_mjlab.sim2sim --mode plane \
  --plane-onnx logs/sim2sim/plane_1600.onnx \
  --jump-onnx logs/sim2sim/jump_2300.onnx
```

点击窗口后按 Enter 开始；Space 切换平地/跳跃策略，1/2 直接选择 plane/jump。
WASD 移动，Q/E/F 调高度，Shift 旋转，X 停止移动，P 暂停，Backspace 重置。
地面为棋盘格并有天空背景，物理接触配置不变；鼠标拖动旋转、滚轮缩放。
切到 jump 不会自动判断落地并切回，需再次按 Space 或按 1。G 暂不支持台阶策略。
导出自己的 checkpoint、无窗口评估、控制接口及验收流程见
[sim2sim 使用说明](docs/sim2sim.md)。

## 固定接口

- `policy`: 当前 25 维本体观测。
- `history`: 单一 25 维 term 的 5 帧历史，按帧排列为 125 维。
- `critic`: 141 维特权观测。
- 动作顺序: `ljoint1, ljoint4, lwheel, rjoint1, rjoint4, rwheel`。
- ONNX: 输入 `obs[B,25]`、`obs_history[B,125]`，输出 `actions[B,6]`。

仿真使用 1 ms Euler/Newton 步长、20 次迭代、`1e-9` 容差；每次 policy step 固定执行 10 个物理子步。腿部在每个子步执行位置 PD，轮子执行速度 PD，两个气弹簧 tendon actuator 的控制量保持为零并由 MJCF bias 产生被动力。

## 文件职责

- `assets/rm26_pnx_wbr_mjcf/`: 当前默认模型，来自指定上游仓库的原始 MJCF 与 15 个 STL。
- `assets/wbr.xml`: 旧的简化模型，仅保留用于历史对照，不再由训练或 play 加载。
- `src/wbr_mjlab/robot.py`: 模型入口和控制系统常量。
- `src/wbr_mjlab/task.py`: plane/jump 共享任务工厂。
- `src/wbr_mjlab/mdp.py`: 状态缓存、观测、混合动作、随机化和课程回调。
- `src/wbr_mjlab/rewards.py`: plane/jump 奖励权重与公式，按名称执行单个分支。
- `src/wbr_mjlab/rl.py`: 最小 Sequence-PPO、续训状态和 ONNX 导出。
- `src/wbr_mjlab/sim2sim.py`: 原生 MuJoCo / ONNX 推理、观测、PD、键盘状态及 CLI。
- `src/wbr_mjlab/sim2sim_viewer.py`: GLFW 键盘窗口与实时调度。
- `tests/test_migration.py`: 模型、MDP、环境、学习与导出验收。
- `assets/policies/legacy_*.onnx`: 仅归档的旧模型，不用于新 WBR 部署或续训。

旧简化模型曾在 RTX 3090 上完成 plane 8192 env、jump 4096 env 的 10 iterations
验收；这些结果不代表当前 mesh 模型已经通过同等规模的训练验收。

## 当前机器人模型

默认使用 [CosmosMount/rm26_pnx_wbr_mjcf](https://github.com/CosmosMount/rm26_pnx_wbr_mjcf)
的 `9c09472` 版本。原始文件及校验值保存在模型目录内，训练与 play 共用同一个加载入口。
不再显示原来的橙色方箱；机身、闭链连杆和轮子均加载完整 STL，地面碰撞也使用这些网格。

加载层保留当前任务的闭链初始姿态和仅与地面碰撞的掩码，避免上游 `home` 位于地下、
以及闭链连接处的网格凸包自碰撞。具体适配记录见
[模型说明](assets/rm26_pnx_wbr_mjcf/README.md)。

腿部电机限幅按上游从 40 Nm 改为 **20 Nm**，轮子仍为 5.2 Nm；旧 checkpoint
的输入输出维度不变，可以加载，但不能保证动作表现不变。建议先评估，再决定续训，
不要把新旧模型的训练结果视为同一物理配置。

本机 RTX 4060 Laptop 已验证：64 个环境下旧 `model_300.pt` 推理 1000 个控制步，
以及 plane/jump 随机动作各 100 步，观测、奖励和状态均为有限值，轮地接触可检测。
旧策略试跑中有 2 次跌倒终止，因此这里只证明能够加载运行，不代表控制性能已达标。
当时的模型迁移回归为 14 项通过，ONNX 对照因缺少 `onnxruntime` 跳过；本次 sim2sim
已补齐 ONNX 对照和原生 MuJoCo 测试，验证范围与结果见上述 sim2sim 使用说明。

已有平地 checkpoint 的查看命令（将路径替换为实际文件）：

```bash
play Mjlab-Velocity-Flat-WBR \
  --checkpoint-file logs/rsl_rl/wbr_plane/<RUN>/model_<ITER>.pt \
  --num-envs 1 --device cuda:0 --viewer native
```

## 性能诊断

关于“几千轮就会行走”的日志证据、迭代与样本量换算，以及本次结构精简，见
[训练分析](docs/training-analysis.md)。其中还记录了跳跃接触力符号导致的腾空误判：
该问题及腾空累计量清零条件现已修复；修复前的跳跃回报不能作为学会起跳的可靠证据。
运行中的训练/播放进程需要重启才能应用修复。旧 checkpoint 仍可加载，但建议新开运行
训练或评估，不要直接比较修复前后的 jump 总回报。

当前完整 mesh 模型在 RTX 4060 Laptop 上、默认 8192 环境、预热 20 步后采样 48 步，
约 **99.2 ms/控制步（8.26 万环境步/秒）**，输出为有限值。它比此前简化模型的
74.4 ms 短测更慢；切换正确外观和碰撞几何并不等于训练加速。这两次测量不是严格
配对实验，也都不包含 PPO 更新。后续若优化碰撞几何，应保留完整视觉网格并重新验收接触行为。

在 GPU 空闲时运行独立采样基准（不训练、不覆盖 checkpoint）：

```bash
python scripts/benchmark_rollout.py --mode plane --num-envs 8192
python scripts/benchmark_rollout.py --mode plane --num-envs 8192 --timings
```

基准使用固定随机动作，预热 20 个控制步后测量 48 步；输出的是环境采样吞吐，
不包含策略推理或 PPO 更新，不能直接与训练日志的 `Perf/total_fps` 比较。
`--timings` 在另外 5 步中对各组件逐一同步，用于区分 GPU 计算与 CPU 等待；
`--profile` 提供 CPU 算子统计，其中拷贝操作耗时可能包含等待此前 GPU 工作的时间。

以下性能数据来自替换前的旧简化模型，不能直接用于当前 mesh 模型。
2026-08-27 在 RTX 4060 Laptop（当前功耗上限 55W）上的短测：8192 个平地环境
原采样约 77.1 ms/控制步，缓存控制常量/索引并去除重复观测与奖励计算后约 74.4 ms，
吞吐约提高 3.7%。同步计时中物理步进约 65.8 ms、额外 `forward` 约 6.35 ms，
主要成本在仿真。该优化不修改步长、求解器、奖励、PPO 或并行环境数量。

可用 `--num-envs`、`--nconmax`、`--njmax`、`--iterations`、`--tolerance`
在基准中独立测试配置，默认训练配置不会被改写。例如 `--nconmax 32 --njmax 128`
的一次短测约 66.9 ms/控制步；这不代表已通过长期稳定性或所有姿态下的容量验收。
减小缓冲区必须检查接触/约束溢出，放宽容差或减少迭代还必须验证闭链误差与训练效果。
本次将容差从 `1e-9` 放宽到 `1e-5` 的短测约 74.1 ms/控制步，与优化后的默认配置
接近，未观察到有意义的提速，因此保留原容差。上述结果是短测，尚未在 RTX 3090 上复测。

默认 `50000` 轮对应 `8192 × 48 × 50000 = 196.6 亿` 条环境转移；若每轮约 4.7 秒，
跑满约需 65 小时。这是预算上限，应结合评估回报与成功率判断是否需要跑满。
