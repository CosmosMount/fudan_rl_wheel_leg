# STM32 USB 硬件在环推理

该流程保留本仓库原生 MuJoCo 的观测、历史、动作缩放和 1 kHz PD，只把 100 Hz
策略网络推理替换为 STM32H723。主机在每个控制步同步发送 `obs + history`，收到并验证
对应 action 后才推进 10 个 MuJoCo 物理子步。

## 安装与启动

USB CDC 串口仅在 HIL 命令启动时加载；训练、导出和普通 sim2sim 不依赖 pyserial：

```bash
micromamba activate wbr
python -m pip install 'pyserial>=3.5,<4'
python -m pip install --no-deps -e .

sim2hil-wbr --port /dev/ttyACM0 --mode plane
```

Ubuntu 上设备通常属于 `root:dialout`。若出现 `Permission denied`，执行
`sudo usermod -aG dialout "$USER"` 后完整注销并重新登录；不要长期使用
`chmod 777 /dev/ttyACM0`，因为权限会在重新枚举后丢失且会允许所有本地用户访问设备。
当前固件的 USB Device 口应显示为 VID:PID `0483:5710`、产品名
`WBR-H723-HIL-v1`；`Doraemon` 表示仍在运行加入 HIL 前的基础固件。不要把 ST-LINK
调试器自带的虚拟串口当成 HIL 端口。CLI 会在握手前打印并检查这个身份。

窗口操作与 sim2sim 相同。Enter 使能，Space 切换 plane/jump，1/2 直接选择策略，
Backspace 重置。握手要求固件同时支持两个模型；切换发生在控制步边界，并按新模式重建
5 帧历史。

无窗口模式默认按墙钟 100 Hz best-effort 运行（普通 Linux 调度并非硬实时），不会因一次
延迟而突发追赶，也不会跳过仿真步：

```bash
sim2hil-wbr --port /dev/ttyACM0 --mode jump --headless \
  --steps 2000 --vx 0.5 --yaw 0.3 --height 0.135 \
  --output logs/hil/jump_eval.npz
```

JSON 结果包含 USB 往返和板端推理的 min/mean/p50/p95/max，以及墙钟 deadline miss
计数。长时间 GUI 会话的 count/min/mean/max 为全程精确统计，p50/p95 来自最多 4096 个
均匀保留样本，内存不会随运行时长增长。NPZ 在 sim2sim 轨迹字段之外保存
`round_trip_us` 和 `inference_us`。调试时可用 `--no-realtime` 尽快运行，但这不能用于
验证 100 Hz 实时能力。

## 固定协议

所有整数和 float32 均为 little-endian。每帧为：

```text
header <IBBHI>: magic=0x314c5257, version=1, type, payload_len, sequence
payload
footer <H>: CRC16(header + payload)
```

CRC 为反射算法，poly `0x8408`、seed `0xffff`、无 final xor。最大 payload 为 608 B。
响应 header 的 sequence 必须回显请求 sequence。

| type | 值 | payload |
| --- | ---: | --- |
| HELLO_REQ | 1 | `<IIII>` session, period_us, model_set_id, flags |
| HELLO_RSP | 2 | `<IIIIIHHHHHH>`，见下方握手约束 |
| INFER_REQ | 3 | `<IB3x150f>` session, mode, obs25, history125 |
| INFER_RSP | 4 | `<IIBBHI6f>` session, input_seq, mode, status, reserved, inference_us, action6 |
| ERROR | 5 | `<IIII>` session, error_code, detail, offending_sequence |

HELLO_REQ 固定为 `period_us=10000`、`model_set_id=0x1c28e40f`、`flags=1`。HELLO_RSP
必须回显 session，并满足 capabilities 含 `0x0f`、同一 model set/period、status=0、
`obs/history/action=25/125/6`、mode mask 含 `0x03`、max payload=608、protocol version=1。

### 模型身份维护

当前固件 model set `0x1c28e40f` 对应的 ONNX 源文件为：

- `plane_15000.onnx`: SHA256
  `d794155543e4529e169b4bf68446f5ef8eef37a6096fad91c3de5d998edf8c0f`
- `jump_25400.onnx`: SHA256
  `fe640156c2df66cc9090834191138c24bed4bad8774e8c392a8c0d07a5c32465`

任一模型重新导出、量化或替换后，都必须选用新的 model-set ID，并同步修改 MCU
`WBR_HIL_MODEL_SET_ID` 与主机 `wbr_mjlab.hil.protocol.MODEL_SET_ID`。否则即使维度相同，
握手也无法防止主机把错权重当成目标策略。

INFER_REQ 的 mode 为 plane=0、jump=1。150 个 float 的顺序是当前 clip 后的 25 维
`obs`，随后是旧帧到新帧排列且不 clip 的 125 维 history。INFER_RSP 的 header sequence
和 payload input_seq 必须同时匹配请求；session、mode、status=0、reserved=0、六个 finite
action 也必须全部通过验证。成功 HELLO 的 sequence 是首个 INFER 的前驱；之后每个请求
必须严格为 `previous + 1`（uint32 回绕）。

## 故障语义

客户端支持串口短读、短写、粘包以及垃圾字节/坏 CRC 后重新寻找 magic，但不会接受
过期、乱序或契约不匹配的响应。初始 HELLO 使用独立的
`--handshake-timeout-ms`（默认 500 ms）；每个 inference 的总写入和读取继续共用
`--timeout-ms`（默认 10 ms）deadline。
每个主机进程使用随机 session 和随机初始 sequence。显式重新握手时，客户端会丢弃可由
sequence 或 session 证明属于旧请求/旧连接的迟到响应；固件把完整且 CRC 正确的
HELLO_REQ 作为同步屏障，因此它也能立即抛弃一次中断写入留下的半帧。

发送前的本地 mode/shape/finite 输入检查失败时不会发包，也不破坏已有握手。任一已发送
inference 的 timeout、串口错误、ERROR、CRC 后无有效帧、序号/session/model/响应 shape
或 finite 检查失败都会 fail closed：本次 MuJoCo 不步进，本次 history append 被回滚，
action、torque 和 ctrl 清零，同时当前握手失效。程序不会继续应用旧 action，也不会自动
含糊重试；恢复前必须重新执行 HELLO。GUI 默认直接安全退出，便于保留首个故障现场。
