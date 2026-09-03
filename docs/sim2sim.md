# mjlab → 原生 MuJoCo / ONNX

目标是先验证部署接口，再验证控制性能。这里直接使用原生 `mujoco.mj_step` 和
ONNX Runtime CPU 推理，不创建训练环境、不运行 PPO、不使用 GPU 物理仿真。
模型仍是项目中的完整 `rm26_pnx_wbr_mjcf`，保留闭链、气弹簧和训练时的碰撞适配。
程序复用项目配置，运行环境仍需安装本项目及其 mjlab 依赖。
画面使用棋盘格平地和渐变天空，便于辨认位移和方向；只增加视觉材质，没有添加
台阶/坡面，也没有改变摩擦或碰撞参数。

## 快速启动

在项目目录、已有 `wbr` 环境运行。ONNX Runtime 已在本机环境安装；其他机器需要：

```bash
micromamba activate wbr
python -m pip install 'onnxruntime>=1.19,<1.24'
```

本机两份示例 ONNX 已导出，推荐一起加载：

```bash
python -m wbr_mjlab.sim2sim --mode plane \
  --plane-onnx logs/sim2sim/plane_1600.onnx \
  --jump-onnx logs/sim2sim/jump_2300.onnx
```

窗口启动时暂停，点击窗口后按 **Enter** 开始；按 **Space** 从平地策略切到跳跃，
完成一次离地和稳定落地后自动切回；**1** 固定选择 plane，**2** 固定选择 jump。
`--mode` 只决定初始策略，默认为 plane；要从跳跃策略开始可设为 `--mode jump`。
两个网络在打开窗口前完成加载和接口校验，切换时不重新加载文件。

导出自己的 checkpoint，`--task` 必须与训练任务相同。只加载自己信任的 `.pt` 文件。
`--verify` 在 64 组固定输入上比较 PyTorch 和 ONNX 的动作，误差超限会报错。

```bash
export-wbr-policy \
  --task Mjlab-Jump-Flat-WBR \
  --checkpoint logs/rsl_rl/wbr_jump/2026-08-27_19-26-43/model_2300.pt \
  --output logs/sim2sim/jump_2300.onnx --verify

python -m wbr_mjlab.sim2sim \
  --mode jump --onnx logs/sim2sim/jump_2300.onnx
```

上面的单策略命令也继续兼容，但只加载 jump 时无法切到 plane；缺失目标策略会在
窗口中提示，不会中断当前策略。可以在该命令中追加 `--plane-onnx` 启用切换。
同一模式不要同时用 `--onnx` 和 `--jump-onnx` 重复指定文件。重新 editable 安装本项目后，也可把
`python -m wbr_mjlab.sim2sim` 换成 `sim2sim-wbr`。

平地策略的导出用 `--task Mjlab-Velocity-Flat-WBR`，运行用 `--mode plane`：

```bash
python -m wbr_mjlab.sim2sim \
  --mode plane --onnx logs/sim2sim/plane_1600.onnx
```

加载静态 MuJoCo XML/MJCF 地形时追加：

```bash
python -m wbr_mjlab.sim2sim \
  --mode plane --onnx logs/sim2sim/plane_1600.onnx \
  --terrain-xml assets/terrains/stairs.xml
```

相对路径以项目根目录为基准。地形可以包含多个 geom 以及 XML 相对路径引用的 mesh、
heightfield、纹理和材质，但不能包含关节或 actuator。地形 geom 的摩擦和碰撞配置来自
XML；匿名 geom 会自动命名，所有可碰撞 geom 都会被轮地接触判断识别。机器人重置位置、
姿态和关节角来自机器人 MJCF 的 `home` keyframe，地形应在该坐标附近提供接触表面。
训练和 mjlab `play` 使用同一文件时设置 `WBR_TERRAIN_XML=assets/terrains/stairs.xml`。

本机也已导出该示例，但它来自较早的 plane checkpoint，不能假设它已适应当前模型。
导出时写入的元数据描述**当前代码的部署约定**，不可能追溯证明旧 checkpoint 的训练
模型、碰撞参数和奖励正确；仍需核对原训练记录。

若当前 shell 导入依赖时出现 `CXXABI` 错误，可在命令前添加
`LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"`。
不要用此变量改变 Python 环境本身，先确认已激活 `wbr`。

## 键盘和窗口

按键参考
[wbr_control 的输入映射](https://github.com/CosmosMount/wbr_control/blob/b3bea37851df7d64395018202d4714157190cc6a/src/input/input_mapper.cpp)、
[指令融合逻辑](https://github.com/CosmosMount/wbr_control/blob/b3bea37851df7d64395018202d4714157190cc6a/src/controller/balance.cpp)
以及 [mujoco_interface 的 GLFW 按键状态](https://github.com/CosmosMount/mujoco_interface/blob/e7eb6cc64644433988779291265e40589c0b78d7/input/src/input.cpp)。
这里只复用操作习惯，不引入 eCAL 或其 LQR/FSM 控制器。

| 按键 | 行为 |
| --- | --- |
| Enter | 切换电机使能；首次按下同时解除暂停 |
| W / S | 按住前进 / 后退，默认 ±0.8 m/s |
| A / D | 按住左转 / 右转，默认 ±1.5 rad/s |
| Q / E / F | 选择低 / 中 / 高机身高度指令 |
| 左 / 右 Shift | 按住原地旋转，释放后恢复 WASD 指令 |
| X | 清空移动键状态、速度指令归零，保留平衡策略 |
| P | 暂停 / 继续物理仿真 |
| Backspace | 恢复初始姿态，清空动作和历史，暂停并解除使能 |
| Esc | 退出 |
| 鼠标左键拖动 / 滚轮 | 旋转相机 / 缩放；相机跟随机身 |
| Space | 触发一次 jump，检测到离地并稳定落地后自动切回 plane |
| 1 / 2 | 直接选择 plane / jump 策略 |
| G | 显示未提供台阶策略的提示，不额外施力 |

松开方向键即取消对应指令；同时按相反方向会抵消。窗口失焦会清空所有移动键，
但策略继续运行以维持平衡。Enter 解除使能时电机输出为零，**物理仿真和气弹簧仍在运行，
机器人可能倒下**；要冻结画面请按 P。普通物理跌倒不会自动重置，方便观察失败。

Q/E/F 对 plane 分别为 `0.10 / 0.15 / 0.20 m`，对 jump 为
`0.12 / 0.135 / 0.15 m`。这是训练中的 **root 高度指令**，不是参考控制器的虚拟腿长。
Shift 映射为策略的 yaw-rate 指令，不切换参考项目的 spin LQR。

运行接口允许的最大指令为 **±3 m/s 前进速度** 与 **±8 rad/s yaw-rate**；命令行可用
`--velocity 3 --yaw-rate 8` 设置键盘幅值，无窗口模式可用 `--vx 3 --yaw 8`。当前已导出的
旧策略是在较窄范围内训练的，只有按新范围重新训练、导出并部署的策略才应被期待稳定覆盖
这些极限指令。

可以调低键盘速度，从小指令开始验收：

```bash
python -m wbr_mjlab.sim2sim --mode jump \
  --onnx logs/sim2sim/jump_2300.onnx --velocity 0.3 --yaw-rate 0.5
```

### 双策略切换的范围

当前策略只有 `[vx, yaw_rate, height]` 三个连续指令，没有 jump/stair 标志或跳跃阶段。
因此 Space 的实现是**临时替换正在运行的 ONNX 策略**，不是给原策略添加 jump 输入，
也不向模型施加冲量。切换在控制步边界生效，同时切换观测缩放、PD 增益和高度范围。
机身姿态、速度、仿真时间以及上一动作保持连续；旧模式的 5 帧历史丢弃，下一次推理
用当前状态按新模式编码并填满历史，避免混用不同的指令缩放。

切换后高度指令回到新模式的中档，WASD 按住状态保留；使能和暂停状态不会改变。
Space 触发后，程序要求连续 2 个控制周期双轮离地，再要求连续 3 个控制周期双轮接地，
然后在下一个控制边界切回 plane。数字键 1/2 仍用于手动选择并会取消一次性跳跃状态；
Backspace 会重置物理状态和一次性跳跃状态。两份策略并未专门联合训练切换过程，建议先
低速/静止触发，不要把能切换理解为任意姿态下都能稳定完成一次跳跃。

## 已对齐的接口

| 项目 | 约定 |
| --- | --- |
| 物理 / 策略周期 | 1 ms / 10 ms，每次动作保持 10 个物理子步 |
| 求解器 | 使用任务配置中的 Euler、Newton、20 次迭代、容差 `1e-9` 等选项 |
| 电机顺序 | `ljoint1, ljoint4, lwheel, rjoint1, rjoint4, rwheel`，按名称索引 |
| 腿部 PD | plane `Kp=20,Kd=1`；jump `Kp=6,Kd=0.5` |
| 腿目标 | `home_q + 0.5 × action`，每个物理子步重新计算力矩 |
| 轮目标 | `10 × action rad/s`，`Kd=0.2` |
| 限幅 | 动作 ±100；腿 ±20 Nm；轮 ±5.2 Nm |
| 气弹簧 | 两个控制输入保持 0，由 MJCF bias 产生被动力 |
| 默认姿态 | 直接来自机器人 MJCF 的 `home` keyframe；修改 XML 后需重启进程 |
| 碰撞 | 保留训练时的轮/连杆网格与地面接触，关闭机器人自碰撞 |
| 推理 | 确定性动作均值，CPU ONNX Runtime 单线程，不采样动作噪声 |

25 维观测依次为：

```text
角速度 × 0.25                 3
机体系重力方向               3
[vx, yaw_rate, height] × scale 3
四个主动腿关节相对 home 位置  4
六个主动关节速度 × 0.05      6
上一策略步动作               6
```

`scale = [2, 0.25, 5]`（plane）或 `[3, 0.25, 5]`（jump）。四元数为 MuJoCo
`wxyz`；角速度与训练一样直接取 free-joint 的旋转 `qvel`，不再旋转一遍。
主动关节速度与当前训练代码一样，使用 **1 ms qpos 差分并 wrap 到 ±π**，不能直接
替换成 `qvel`，也不能在策略周期上做差分。

ONNX 输入为 `obs[1,25]` 和 `obs_history[1,125]`。历史按**旧帧 → 新帧**排列，
包含当前帧；初始时用第一帧填满 5 帧。当前帧 clip 到 ±100，历史不 clip。
这是现有训练接口，部署端没有自行改成全零历史、反向历史或额外归一化。

新导出文件记录任务、关节顺序、增益、步长和 MJCF 哈希，程序拒绝不匹配的元数据。
旧 ONNX 没有元数据时会警告，仍需人工确认 `--mode` 和训练配置。
`assets/policies/legacy_*.onnx` 是归档，不应作为当前模型的部署策略。

## 建议的 sim2sim 验收顺序

1. **先确认模型与 checkpoint 配对。** 当前完整模型与旧简化模型不同；闭链、气弹簧、
   质量惯量、碰撞、关节符号和电机限幅都要核对。先在 mjlab 自身确认策略能够跟踪指令。
2. **隔离网络导出。** 对同一批真实观测分别运行 `.pt` 与 ONNX，比较动作。不要先比较
   两段闭环轨迹，因为微小物理差异会随时间放大。`--verify` 是随机输入的快速检查，
   实际轨迹的观测可按下面方式保存后复核。
3. **隔离控制接口。** 关闭随机化和噪声，使用相同初始姿态/速度和固定指令，比较第一帧
   观测、5 帧历史，以及相同状态/动作下每个电机的 PD 力矩。mjlab 的默认 play 重置仍有
   随机根速度；做严格比较时需在独立评估配置中去掉 `reset_velocity`。
4. **再测闭环。** 顺序测试零速度、±0.2/±0.5 m/s、左右转向、停止与高度变化；统计速度
   RMSE、倾角、跌倒、力矩饱和与接触。两边能运行或没有 NaN，不代表指令跟踪合格。
5. **最后增加部署差异。** 逐项引入 IMU/编码器噪声、延迟、控制抖动、摩擦、质量偏差，
   再做对应训练随机化；不要在第一轮同时更换控制器、网络和碰撞模型。

mjlab 这里使用 MuJoCo Warp，当前程序使用原生 MuJoCo。因此它能检查部署数据通路和
后端差异，但仍属于 MuJoCo 系列，不能替代真机动力学、传感器和通信验证。
UI 慢时程序让仿真慢于真实时间，不修改 dt、不跳过 PD 或策略步。

## 无窗口评估与轨迹

```bash
python -m wbr_mjlab.sim2sim --mode jump \
  --onnx logs/sim2sim/jump_2300.onnx \
  --headless --steps 2000 --vx 0.5 --yaw 0.3 --height 0.135 \
  --output logs/sim2sim/eval.npz
```

2000 步是 20 秒仿真时间，不是 PPO 更新次数。无窗口模式自动使能并尽快运行，不需要
显示器。输出包含耗时、最小根高度、最终位置；`finite=true` 仅表示未出现数值异常。
NPZ 保存 `time/qpos/qvel/command/obs/history/action/torque` 和 JSON metadata。
每行 `obs/history/action` 属于该控制步的输入与动作，`qpos/qvel/time` 属于执行后的状态，
`torque` 是最后一个物理子步的电机力矩，不是整步平均值。

实现分为 `sim2sim.py`（ONNX、观测、PD、按键状态和 CLI）与
`sim2sim_viewer.py`（GLFW 渲染与时钟）。核心测试不依赖显示器；GUI 冒烟测试使用
自动调用已注册按键回调的方式，不等于人工完成所有操作验收。

## 本机初步验证（2026-08-27）

使用 plane `2026-08-27_11-43-41/model_1600.pt`、jump
`2026-08-27_19-26-43/model_2300.pt`，没有修改 checkpoint，也没有启动训练。
两种 ONNX 分别完成原生 MuJoCo 20 秒运动试跑，未出现数值异常。另做相同起点、
固定 `vx=0.5 m/s, yaw=0.3 rad/s` 的 10 秒对照；关闭 mjlab 重置速度随机化，
统计时去掉前 2 秒过渡。这里的 mjlab 对照使用 **CPU 上的 MuJoCo Warp**，尚未做
GPU 后端的同条件重复实验。

| 指标 | plane：mjlab / 原生 | jump：mjlab / 原生 |
| --- | --- | --- |
| 平均机体系前向速度（m/s） | 0.468 / 0.474 | 0.507 / 0.551 |
| 前向速度 RMSE（m/s） | 0.0335 / 0.0283 | 0.0286 / 0.0562 |
| 转速 RMSE（rad/s） | 0.0231 / 0.0252 | 0.0644 / 0.0552 |
| 2000 组实际观测的 Torch/ONNX 最大动作误差 | 5.36e-6 | 4.56e-6 |

初始观测和历史逐元素一致，检查过的碰撞、质量惯量、执行器参数没有差异。
结果保存在 `logs/sim2sim/validation.json`；初版 GUI 截图为 `logs/sim2sim/viewer.png`。
这是有限场景的初测，不证明全速度范围、抗扰动、按指令跳跃或真机部署已合格。
尤其 jump 的后端差异需要进一步评估。转向试验会绕圈，不能用最终位移除以时间来
判断前向速度跟踪，必须在机体系中统计。

后续修复了鼠标回调对 `mjv_moveCamera` 多传 `scene` 的问题。回归测试直接调用当前
MuJoCo 的相机函数，覆盖拖动、双向滚轮、模式切换时的物理状态连续性、新模式的观测
历史和 PD 力矩、单/双策略命令行。新增窗口检查还包括运行中切换、暂停时切换和地面
材质渲染；截图保存为 `logs/sim2sim/viewer_switch_terrain.png`。
