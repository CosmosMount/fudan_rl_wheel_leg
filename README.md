> **首先安装 Isaac Gym：前往 <https://developer.nvidia.com/isaac-gym> 下载 Isaac Gym Preview 4，解压后进入 `isaacgym/python`，执行 `pip install -e .`。**


在本项目中：
- `plane`：平地运动(但是有地形)。
- `jump`：跳跃任务。

## 1. 环境要求

- Linux(我用的Ubuntu22.04)   在gpufree算力自由平台上租用的服务器  感谢算力自由平台！(广告位招租)
- NVIDIA GPU 与可用的 NVIDIA 驱动
- CUDA 兼容的 PyTorch
- Conda
- Python 3.8
- Isaac Gym Preview 4

(windows似乎可以用wsl，但俺没试过

创建新环境：

```bash
conda create -n leg python=3.8 -y
conda activate leg
```

## 2. 安装 Isaac Gym

1. 前往 <https://developer.nvidia.com/isaac-gym> 下载 Isaac Gym Preview 4。
2. 解压后安装 Python 包：

安装教程可以参考这篇文章，提出了numpy和matp库版本问题解决方法。
https://blog.csdn.net/littlewells/article/details/140179837

```bash
cd ~/isaacgym/python
pip install -e .
```

3. 运行 NVIDIA 示例验证安装：

```bash
cd ~/isaacgym/python/examples
python 1080_balls_of_solitude.py
```

正常显示大量小球即说明 Isaac Gym 基本可用。若提示段错误，则考虑显卡驱动版本和设备是否有显示器，若无显示器则只能使用headless训练

如果出现 `libpython3.8m.so.1.0` 或动态库找不到的问题，可先执行：

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

## 3. 安装本项目

### 3.1 平地与通用运动版本

```bash
cd ~/plane
pip install -e .
```

### 3.2 跳跃版本

```bash
cd ~/jump
pip install -e .
```
## 4. 在isaacgym中train和play

```bash
tensorboard --logdir logs --port 8080
```
浏览器打开：
```text
http://localhost:8080
```

需要先解压meshes文件
```bash
cd ~/plane/resources/robots/infantry_V4
unzip meshes.zip

cd ~/jump/resources/robots/infantry_V4
unzip meshes.zip
```
train和play脚本见isaacgym脚本.txt

## 5.在mujoco中进行sim2sim验证

新创建一个python3.9的环境，mujoco版本为3.2.7，其他的缺啥补啥即可
```bash
cd mujoco
python python_tools/onnx_mj_chuanlian.py
python python_tools/onnx_mj_binglian.py
```
需要有桌面的或者通过X11转发进行可视化

## 6. 致谢
本项目基于以下工作：

感谢玺佬开源感谢玺佬开源！！！！！！

- [玺佬万岁万岁万万岁](https://github.com/clearlab-sustech/Wheel-Legged-Gym.git)

以及：
- [NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym)
- [legged_gym](https://github.com/leggedrobotics/legged_gym)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)

感谢chatgpt、codex等ai提供的大力支持！

代码简读见bbs
