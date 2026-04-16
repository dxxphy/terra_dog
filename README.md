# TERRA_DOG

## 项目名称：面向复杂地形的轮足式机器狗运动控制研究

### 项目概述

结合轮式机器人的效率与足式机器人的越障能力，本项目研究轮足式机器狗在复杂地形（如楼梯、碎石、斜坡）下的混合运动模式。核心内容包括：基于强化学习的全地形运动控制，旨在提升机器人在非结构环境下的通过性和稳定性。

本仓库强化学习部分基于：

- [N3PO Locomotion](https://github.com/zeonsunlightyu/LocomotionWithNP3O.git)
- [TITATIT-Quadruped-Wheeled Mode](https://github.com/DDTRobot/quadruped-wheel-titatit-rl)

### 参考环境

| Environment        | Description                     |
| ------------------ | ------------------------------- |
| 显卡               | RTX 4090                        |
| CUDA               | CUDA 12.4                       |
| 训练环境           | Isaac Gym                       |
| sim2sim            | Webots 2023                     |
| ROS                | ROS 2 Humble                    |
| 推理               | RTX 4090 / Jetson Orin NX on TITA + TensorRT |
| 虚拟环境           | Miniconda                       |

---

## 快速开始

### 项目结构

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
└── logs/                # 训练日志和模型保存目录
```

---

## 训练

### 开始训练

使用默认配置开始训练：

```bash
python train.py --task=wheeled_titatit
```

### 自定义训练参数

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

### 从某个位置继续训练

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

### 训练参数说明

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

### 使用 TensorBoard

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

### 理解关键指标

- **episode_length**: 每个回合的平均长度
- **mean_reward**: 平均奖励（越高越好）
- **mean_episode_length**: 平均回合长度
- **mean_episode_reward**: 平均回合奖励
- **action_rate**: 动作变化率（越小越平滑）
- **z_vel**: Z轴速度（垂直运动，越小越稳定）
- **xy_vel**: XY角速度（旋转运动，越小越稳定）
- **feet_air_reward**: 脚在空中的奖励

### 查看保存的模型

训练过程中会定期保存模型 checkpoint，保存在 `logs/rough_wheeled_titatit_constraint/<run_name>/` 目录下：

```bash
# 列出所有运行
ls -lh logs/rough_wheeled_titatit_constraint/

# 列出特定运行的所有模型
ls -lh logs/rough_wheeled_titatit_constraint/Apr07_01-51-37_test_barlowtwins_feetcontact/
```

模型文件格式为 `model_<iteration>.pt`，例如 `model_1000.pt`、`model_3000.pt` 等。

---

## 模型测试与可视化

### Simple Play 基本用法

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


### 推荐工作流程

1. **使用 TensorBoard 查看训练曲线**
   ```bash
   tensorboard --logdir logs/rough_wheeled_titatit_constraint
   ```
   在浏览器中打开 `http://localhost:6006` 查看训练曲线，找出性能最好的 checkpoint。

2. **加载性能最好的模型**
   假设发现 3500 步的奖励最高：
   ```bash
   python simple_play.py --task=wheeled_titatit \
       --checkpoint_path logs/rough_wheeled_titatit_constraint/Apr07_01-51-37_test_barlowtwins_feetcontact/model_3500.pt
   ```

3. **查看生成的统计信息**
   运行后会在终端输出：
   - `action rate`: 动作变化率（越小越平滑）
   - `z vel`: Z轴速度（垂直运动，越小越稳定）
   - `xy_vel`: XY角速度（旋转运动，越小越稳定）
   - `feet air reward`: 脚在空中的奖励
   
   这些指标越低，说明机器人运动越平稳。

4. **录制视频**
   运行后会在当前目录生成 `record.mp4` 文件。

### Simple Play 参数说明

| 参数              | 类型    | 默认值    | 说明                                   |
| ----------------- | ------- | --------- | -------------------------------------- |
| `--task`          | str     | wheeled_titatit | 任务名称                           |
| `--checkpoint_path` | str   | None      | 模型文件的完整路径                     |
| `--headless`      | flag    | False     | 无头模式（不显示图形界面）             |

### 视频录制说明

- 默认使用 MP4V 编码器，生成的视频格式为 MP4
- 视频保存在当前目录，文件名为 `record.mp4`
- 视频时长约 40 秒（可在 `simple_play.py` 第 97 行修改）
- 如果需要在某些播放器中播放，可能需要安装相应的解码器

**更换视频编码器：**

如需更通用的格式，可以修改 `simple_play.py` 第 130 行：

```python
# 改为 MJPEG（最兼容）
video = cv2.VideoWriter('record.avi', cv2.VideoWriter_fourcc(*'MJPG'), int(1 / env.dt), (img.shape[1],img.shape[0]))

# 改为 H.264（需要额外编译支持）
video = cv2.VideoWriter('record.mp4', cv2.VideoWriter_fourcc(*'avc1'), int(1 / env.dt), (img.shape[1],img.shape[0]))
```

---

## 高级功能

### 批量生成多个 checkpoint 的视频

创建一个脚本 `batch_record.sh`：

```bash
#!/bin/bash
RUN_NAME="Apr07_01-51-37_test_barlowtwins_feetcontact"
CHECKPOINTS=(1000 2000 3000 4000 5000 6000)

for i in "${CHECKPOINTS[@]}"; do
    echo "Processing model_$i.pt..."
    python simple_play.py --task=wheeled_titatit \
        --checkpoint_path logs/rough_wheeled_titatit_constraint/$RUN_NAME/model_${i}.pt
    mv record.mp4 record_model_${i}.mp4
done

echo "All videos generated!"
```

运行脚本：

```bash
chmod +x batch_record.sh
./batch_record.sh
```

### 导出模型为 TorchScript 格式

训练好的模型可以导出为 TorchScript 格式，用于部署：

```bash
python simple_play.py --task=wheeled_titatit --checkpoint_path model_6000.pt
```

运行后会自动在当前目录生成 `model.pt` 文件，这是导出的 TorchScript 模型。

---

## 常见问题

### Q: 如何知道有哪些模型可以加载？

```bash
# 列出所有运行
ls -lh logs/rough_wheeled_titatit_constraint/

# 列出特定运行的所有模型
ls -lh logs/rough_wheeled_titatit_constraint/Apr07_01-51-37_test_barlowtwins_feetcontact/
```

### Q: 训练时出现 CUDA out of memory 错误怎么办？

减少环境数量：

```bash
python train.py --task=wheeled_titatit --num_envs=2048
```

或者使用 CPU 训练（速度会很慢）：

```bash
python train.py --task=wheeled_titatit --rl_device=cpu
```


### Q: Simple Play 生成的视频无法播放怎么办？

尝试更换视频编码器（参见"视频录制说明"部分），或使用其他播放器（如 VLC）。

### Q: 如何调整机器人的运动命令？

在 `simple_play.py` 第 117-120 行修改命令：

```python
env.commands[:,0] = 1  # x方向速度
env.commands[:,1] = 0  # y方向速度
env.commands[:,2] = 0  # 旋转速度
env.commands[:,3] = 0  # 步态类型
```

### Q: 训练多久可以看到效果？

通常需要训练 2000-3000 次迭代才能看到初步效果，完整训练可能需要 6000-10000 次迭代。具体时间取决于：

- 硬件性能（GPU）
- 环境数量
- 任务复杂度

建议使用 TensorBoard 实时监控训练进度。

### Q: 如何使用多个 GPU 进行训练？

使用 Horovod 进行多 GPU 训练：

```bash
python train.py --task=wheeled_titatit --horovod
```

---

## 项目说明

### 文件说明

- `train.py`: 主训练脚本
- `simple_play.py`: 模型测试和可视化脚本
- `global_config.py`: 全局配置
- `configs/`: 任务和算法配置文件
- `envs/`: 环境实现
- `modules/`: 神经网络模块
- `algorithm/`: RL 算法实现
- `runner/`: 训练运行器
- `utils/`: 工具函数
- `resources/`: 机器人模型资源文件

### 配置文件

主要的配置文件位于 `configs/` 目录：

- `wheeled_titatit_constraint_him.py`: TitaTIT 轮足机器人配置

可以修改这些文件来调整：

- 奖励函数
- 环境参数
- 地形配置
- 网络结构
- 训练超参数

### 日志和模型

训练日志和模型保存在 `logs/` 目录下：

```
logs/
└── rough_wheeled_titatit_constraint/
    ├── <run_name_1>/
    │   ├── model_1000.pt
    │   ├── model_2000.pt
    │   ├── ...
    │   └── events.out.tfevents...
    └── <run_name_2>/
        └── ...
```

---

## 参考资源

- [Isaac Gym Documentation](https://developer.nvidia.com/isaac-gym)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [TensorBoard Documentation](https://www.tensorflow.org/tensorboard)
- [N3PO Locomotion](https://github.com/zeonsunlightyu/LocomotionWithNP3O.git)
- [TITATIT-Quadruped-Wheeled Mode](https://github.com/DDTRobot/quadruped-wheel-titatit-rl)

---

## 许可证

请参考 [LICENSE](LICENSE) 文件。

---

## 联系方式

如有问题或建议，请通过 GitHub Issues 联系。