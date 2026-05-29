# TERRA_DOG

## 项目名称：面向复杂地形的轮足式机器狗运动控制研究

### 项目概述

结合轮式机器人的效率与足式机器人的越障能力，本项目研究轮足式机器狗在复杂地形（如楼梯、碎石、斜坡）下的混合运动模式。核心内容包括：基于强化学习的全地形运动控制，旨在提升机器人在非结构环境下的通过性和稳定性。

本仓库强化学习部分基于：

- [N3PO Locomotion](https://github.com/zeonsunlightyu/LocomotionWithNP3O.git)
- [TITATIT-Quadruped-Wheeled Mode](https://github.com/DDTRobot/quadruped-wheel-titatit-rl)v

---

### 参考环境

| Environment        | Description                                  |
| ------------------ | -------------------------------------------- |
| 显卡               | RTX 4090                                     |
| CUDA               | CUDA 12.4                                    |
| 训练环境           | Isaac Gym                                    |
| sim2sim            |MuJoCo                                       |
| ROS                | ROS 2 Humble                                 |
| 推理               | RTX 4090 / Jetson Orin NX on TITA + TensorRT |
| 虚拟环境           | Miniconda (Python 3.8)                       |

---

## 项目结构

```
quadruped-wheel-titatit-rl/
├── algorithm/           # 算法实现（PPO等）
├── configs/             # 配置文件
│   └── wheeled_titatit_constraint_him.py
├── envs/                # 环境定义
│   ├── legged_robot.py  # 腿式机器人基类
│   └── vec_env.py       # 向量化环境
├── modules/             # 神经网络模块
│   ├── actor_critic.py  # Actor-Critic网络
│   └── common_modules.py
├── runner/              # 训练运行器
├── utils/               # 工具函数
├── resources/           # 资源文件（URDF、mesh等）
├── train.py             # 训练脚本
├── simple_play.py       # 模型测试/可视化脚本
├── ddt_ros2_ws/         # sim2sim仿真环境（Git Submodule）
└── logs/                # 训练日志和模型保存目录
```

---

## 环境搭建

### 1. 安装 NVIDIA 显卡驱动

使用 Ubuntu 软件中心安装或运行以下命令：

```bash
sudo apt update
sudo apt install nvidia-driver-535
```

### 2. 安装 Miniconda

下载并安装 Miniconda，然后创建项目专用虚拟环境。

```bash
# 1. 下载 Miniconda 安装脚本
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 2. 运行安装脚本（阅读许可协议，输入 yes 同意，按回车确认默认安装路径）
bash Miniconda3-latest-Linux-x86_64.sh

# 3. 初始化 conda（安装完成后运行，或重启终端）
source ~/.bashrc

# 4. 创建 Python 3.8 虚拟环境
conda create -n terra_dog python=3.8 -y

# 5. 激活虚拟环境
conda activate terra_dog
```

### 3. 安装 CUDA 与 PyTorch

本项目使用 PyTorch 自带的 CUDA 运行时库，无需手动安装完整的 CUDA Toolkit。

#### 3.1 验证驱动

使用以下指令检查驱动是否正常安装：

```bash
nvidia-smi
```

#### 3.2 安装 PyTorch (CUDA 12.4)

在激活的虚拟环境中安装 PyTorch：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

#### 3.3 验证 CUDA 版本

安装后使用以下指令检查是否正常运行 CUDA 12.4：

```bash
python -c "import torch; print(torch.version.cuda)"
```

> **注意**：如需完整的 CUDA Toolkit（如用于编译自定义 CUDA 算子），请参考 [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) 进行安装。

### 4. 安装 TensorRT

根据 CUDA 版本安装对应的 TensorRT。本项目使用 CUDA 12.4，对应安装 TensorRT 10.13.0.35 版本。

参考链接：[NVIDIA TensorRT Download](https://developer.nvidia.com/nvidia-tensorrt-10x-download)

### 5. 安装 Isaac Gym

参考链接：[Isaac Gym Download](https://developer.nvidia.com/isaac-gym/download)

> **重要提示**：  
> Isaac Gym 官方支持到 Ubuntu 20.04 和 Python 3.8。如果在 Ubuntu 22.04 上遇到兼容性问题（通常是因为系统 Python 默认为 3.10），请参考以下解决方案：
>
> [Isaac Gym 在 Ubuntu 22.04 上的安装问题解决方法](https://blog.csdn.net/sinat_27236401/article/details/148376823)

---

## 训练指南

### 1. 基本训练

使用默认配置开始训练：

```bash
python train.py --task=wheeled_titatit
```

### 2. 自定义训练参数

```bash
# 指定训练迭代次数
python train.py --task=wheeled_titatit --max_iterations=10000

# 指定环境数量
python train.py --task=wheeled_titatit --num_envs=4096

# 设置随机种子
python train.py --task=wheeled_titatit --seed=42

# 指定实验名称
python train.py --task=wheeled_titatit --experiment_name=my_experiment

# 指定运行名称
python train.py --task=wheeled_titatit --run_name=test_run
```

### 3. 断点续训

从最新的 checkpoint 继续训练：

```bash
python train.py --task=wheeled_titatit --resume
```

从指定的运行继续训练：

```bash
# 继续上一次运行
python train.py --task=wheeled_titatit --resume --load_run=-1

# 继续指定的运行（使用运行目录名）
python train.py --task=wheeled_titatit --resume --load_run=Apr07_01-51-37_test_barlowtwins_feetcontact

# 从指定的 checkpoint 继续
python train.py --task=wheeled_titatit --resume --checkpoint=3000

# 组合使用
python train.py --task=wheeled_titatit --resume --load_run=Apr07_01-51-37_test_barlowtwins_feetcontact --checkpoint=5000
```

### 4. 训练参数说明

| 参数                   | 类型    | 默认值    | 说明                                     |
| ---------------------- | ------- | --------- | ---------------------------------------- |
| `--task`               | str     | go1       | 任务名称                                 |
| `--resume`             | flag    | False     | 是否从 checkpoint 继续训练               |
| `--experiment_name`    | str     | -         | 实验名称                                 |
| `--run_name`           | str     | -         | 运行名称                                 |
| `--load_run`           | str     | -1        | 要加载的运行名称，-1 表示最新运行        |
| `--checkpoint`         | int     | -1        | checkpoint 编号，-1 表示最新 checkpoint  |
| `--max_iterations`     | int     | -         | 最大训练迭代次数                         |
| `--num_envs`           | int     | -         | 环境数量                                 |
| `--seed`               | int     | -         | 随机种子                                 |
| `--headless`           | flag    | False     | 无头模式（不显示图形界面）               |
| `--rl_device`          | str     | cuda:0    | RL 算法使用的设备                        |

---

## 监控训练

### 1. 使用 TensorBoard

在训练过程中或训练后，使用 TensorBoard 查看训练曲线和指标：

```bash
# 启动 TensorBoard
tensorboard --logdir logs/rough_wheeled_titatit_constraint

# 指定端口
tensorboard --logdir logs/rough_wheeled_titatit_constraint --port 6007

# 允许外部访问
tensorboard --logdir logs/rough_wheeled_titatit_constraint --bind_all
```

然后在浏览器中打开 `http://localhost:6006` 查看训练曲线。

### 2. 理解关键指标

- **episode_length**: 每个回合的平均长度
- **mean_reward**: 平均奖励（越高越好）
- **mean_episode_length**: 平均回合长度
- **mean_episode_reward**: 平均回合奖励
- **action_rate**: 动作变化率（越小越平滑）
- **z_vel**: Z轴速度（垂直运动，越小越稳定）
- **xy_vel**: XY角速度（旋转运动，越小越稳定）
- **feet_air_reward**: 脚在空中的奖励

---

## 模型测试与可视化

### 1. Simple Play 基本用法

`simple_play.py` 用于加载训练好的模型并进行可视化测试。

#### 加载默认模型

```bash
python simple_play.py --task=wheeled_titatit
```

这会加载根目录下的 `model_6000.pt`（默认模型）。

#### 加载指定路径的模型

```bash
# 使用相对路径
python simple_play.py --task=wheeled_titatit \
    --checkpoint_path logs/rough_wheeled_titatit_constraint/Apr07_01-51-37_test_barlowtwins_feetcontact/model_3000.pt

# 加载根目录的模型
python simple_play.py --task=wheeled_titatit --checkpoint_path model_6000.pt
```

#### 无头模式（不显示图形界面）

```bash
python simple_play.py --task=wheeled_titatit --headless
```

### 2. Simple Play 参数说明

| 参数              | 类型    | 默认值    | 说明                                   |
| ----------------- | ------- | --------- | -------------------------------------- |
| `--task`          | str     | wheeled_titatit | 任务名称                           |
| `--checkpoint_path` | str   | None      | 模型文件的完整路径                     |
| `--headless`      | flag    | False     | 无头模式（不显示图形界面）             |

---

## 参考资源

- [Isaac Gym Documentation](https://developer.nvidia.com/isaac-gym)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [TensorBoard Documentation](https://www.tensorflow.org/tensorboard)
- [N3PO Locomotion](https://github.com/zeonsunlightyu/LocomotionWithNP3O.git)
- [TITATIT-Quadruped-Wheeled Mode](https://github.com/DDTRobot/quadruped-wheel-titatit-rl)

---
