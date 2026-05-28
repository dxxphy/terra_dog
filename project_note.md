# Quadruped-Wheel-Titatit-RL 项目文档

## 项目概述

这是一个基于Isaac Gym的轮式四足机器人（Titati）强化学习项目，使用约束PPO（NP3O）算法进行训练。该项目实现了在复杂地形上行走的机器人控制策略，包含了仿真环境、神经网络模型、训练算法和完整的工具链。

---

## 项目结构

```
quadruped-wheel-titatit-rl/
├── algorithm/              # 算法实现
├── configs/                # 配置文件
├── envs/                   # 环境实现
├── logs/                   # 训练日志和模型保存
├── modules/                # 神经网络模块
├── resources/              # 资源文件（URDF、Mesh等）
├── runner/                 # 训练运行器
├── utils/                  # 工具函数
├── global_config.py         # 全局配置
├── train.py                # 训练入口
├── simple_play.py          # 测试/演示脚本
└── README.md               # 项目说明
```

---

## 核心文件详解

### 1. 根目录文件

#### 1.1 global_config.py
**作用**: 全局配置文件，定义项目路径常量

**内容**:
- `ROOT_DIR`: 项目根目录路径
- `ENVS_DIR`: 环境目录路径（当前未使用）

**关键代码**:
```python
ROOT_DIR = os.path.dirname(__file__)
ENVS_DIR = os.path.join(ROOT_DIR,'Env')
```

#### 1.2 train.py
**作用**: 训练主程序入口

**功能**:
- 解析命令行参数
- 创建仿真环境
- 初始化算法和策略
- 执行训练循环
- 保存训练模型

**主要流程**:
1. 注册任务到TaskRegistry
2. 创建环境和算法运行器
3. 调用runner.learn()开始训练
4. 定期保存模型

#### 1.3 simple_play.py
**作用**: 测试和演示脚本，用于加载训练好的模型进行推理

**功能**:
- 加载预训练模型
- 在环境中运行策略
- 可视化机器人行为
- 支持不同的测试场景

---

### 2. algorithm/ 目录

#### 2.1 algorithm/np3o.py
**作用**: NP3O（Near-Pareto Optimal PPO）算法实现

**核心功能**:
- PPO算法的变体，支持成本约束
- 约束违反项的计算
- 价值函数和成本价值函数的训练
- 模仿学习支持（通过BarlowTwins损失）

**主要类和方法**:

**NP3O类**:
- `__init__()`: 初始化算法参数，包括gamma、lambda、clip参数等
- `init_storage()`: 初始化经验回放缓冲区
- `act()`: 根据观测选择动作
- `process_env_step()`: 处理环境步骤，存储转换
- `compute_returns()`: 计算奖励回报
- `compute_cost_returns()`: 计算成本回报
- `compute_surrogate_loss()`: 计算PPO surrogate损失
- `compute_cost_surrogate_loss()`: 计算成本surrogate损失
- `compute_viol()`: 计算约束违反损失
- `update()`: 更新策略网络
- `update_k_value()`: 动态更新约束系数k

**关键特性**:
- 支持多个成本函数
- 动态调整约束权重k_value
- 可选的模仿学习损失
- 支持RNN和前馈网络

---

### 3. configs/ 目录

#### 3.1 configs/base_config.py
**作用**: 基础配置类，定义所有配置项的默认值

**主要配置类别**:
- **sim**: 仿真器参数（物理引擎、时间步等）
- **env**: 环境参数（观察空间、动作空间等）
- **terrain**: 地形参数
- **commands**: 命令参数（速度、方向等）
- **control**: 控制参数（PD控制器、动作缩放等）
- **asset**: 机器人资产参数（URDF文件、碰撞等）
- **rewards**: 奖励函数配置
- **normalization**: 观测和动作归一化
- **domain_rand**: 域随机化参数
- **noise**: 观测噪声配置
- **viewer**: 可视化配置

#### 3.2 configs/legged_robot_config.py
**作用**: 腿式机器人的具体配置类

**主要配置**:
- 继承自BaseConfig
- 定义机器人特定的奖励函数
- 设置默认关节角度
- 配置地形类型和难度

#### 3.3 configs/wheeled_titatit_constraint_him.py
**作用**: 轮式Titati机器人的约束强化学习配置

**特定配置**:
- 机器人资产路径
- 动作空间维度（16个关节）
- 观察空间配置
- 奖励函数权重
- 成本函数约束（如关节限制、碰撞等）
- 模仿学习参数（imi_flag, teacher_act）
- BarlowTwins损失参数

---

### 4. envs/ 目录

#### 4.1 envs/base_task.py
**作用**: Isaac Gym环境的基类

**功能**:
- 封装Isaac Gym的API
- 提供环境创建、重置、步骤等基础接口
- 管理仿真器实例

#### 4.2 envs/legged_robot.py
**作用**: 腿式机器人环境的核心实现

**核心类**: `LeggedRobot`

**主要功能**:

**初始化** (`__init__`):
- 解析配置文件
- 创建仿真和地形
- 初始化PyTorch缓冲区
- 准备奖励和成本函数

**环境管理**:
- `_create_envs()`: 创建多个并行环境实例
- `create_sim()`: 创建仿真器、地形和环境
- `_create_heightfield()`: 创建高度场地形
- `_create_trimesh()`: 创建三角形网格地形

**观测计算**:
- `compute_observations()`: 计算机器人观测
- 观测包括：角速度、重力投影、命令、关节位置、关节速度、历史观测、接触状态、高度测量等

**奖励系统**:
- `_prepare_reward_function()`: 准备奖励函数列表
- `compute_reward()`: 计算总奖励
- 支持多种奖励项：跟踪速度、站立稳定性、关节平滑性、接触奖励等

**成本函数**:
- `_prepare_cost_function()`: 准备成本函数
- `compute_cost()`: 计算成本（约束违反程度）
- 包括：关节位置限制、关节速度限制、关节扭矩限制等

**动作执行**:
- `_compute_torques()`: 将动作转换为关节扭矩
- 支持PD控制和直接扭矩控制
- 包含扭矩限制和低通滤波

**重置机制**:
- `reset_idx()`: 重置指定环境
- `_reset_dofs()`: 重置关节位置和速度
- `_reset_root_states()`: 重置机器人基座状态

**域随机化**:
- `_process_rigid_shape_props()`: 随机化物理属性（摩擦力、恢复系数）
- `_process_rigid_body_props()`: 随机化质量和质心
- `_process_dof_props()`: 随机化PD增益
- 支持电机强度随机化
- 支持动作延迟模拟

**其他功能**:
- `check_termination()`: 检查是否需要重置
- `_resample_commands()`: 重新采样目标命令
- `_push_robots()`: 随机推动机器人增加鲁棒性
- `set_camera()`: 设置相机视角

**关键数据结构**:
- `root_states`: 机器人根状态（位置、姿态、线速度、角速度）
- `dof_pos`: 关节位置
- `dof_vel`: 关节速度
- `contact_forces`: 接触力
- `obs_buf`: 观测缓冲区
- `priv_obs_buf`: 特权观测（包含额外信息）
- `actions`: 动作
- `torques`: 关节扭矩

#### 4.3 envs/vec_env.py
**作用**: 向量化环境包装器

**功能**:
- 管理多个并行环境实例
- 提供统一的接口
- 支持批量操作

---

### 5. modules/ 目录

#### 5.1 modules/actor_critic.py
**作用**: Actor-Critic神经网络架构

**主要类**:

**ActorCriticBarlowTwins类**:
- Actor网络：输出动作均值
- Critic网络：输出价值估计
- Cost网络：输出成本估计
- 支持模仿学习（BarlowTwins损失）

**网络结构**:

1. **Actor Teacher Backbone** (`MlpBarlowTwinsActor`):
   - 历史观测编码器（MLP with LayerNorm）
   - 当前观测编码器
   - 动作生成网络
   - BarlowTwins损失计算（用于自监督学习）

2. **编码器**:
   - `priv_encoder`: 特权信息编码器
   - `scan_encoder`: 扫描点编码器
   - `history_encoder`: 历史观测编码器（CNN）

3. **价值网络**:
   - 输入：当前观测 + 特权信息 + 历史编码
   - 输出：标量价值

4. **成本网络**:
   - 输入：当前观测 + 特权信息 + 历史编码
   - 输出：多个成本值（Softplus激活）

**关键方法**:
- `act()`: 采样动作
- `evaluate()`: 评估状态价值
- `evaluate_cost()`: 评估成本
- `get_actions_log_prob()`: 获取动作对数概率
- `imitation_learning_loss()`: 计算模仿学习损失
- `save_torch_jit_policy()`: 导出TorchScript模型

**输入观测结构**:
```
obs = [
    proprioceptive_obs,  # 本体感知（48维）
    privileged_obs,      # 特权信息（质量、摩擦力等）
    history_obs          # 历史观测
]
```

#### 5.2 modules/common_modules.py
**作用**: 通用神经网络模块

**主要组件**:

1. **激活函数** `get_activation()`:
   - 支持ELU, SELU, ReLU, LeakyReLU, Tanh, Sigmoid等

2. **MLP工厂函数**:
   - `mlp_factory()`: 创建标准MLP
   - `mlp_layernorm_factory()`: 创建带LayerNorm的MLP

3. **状态历史编码器** `StateHistoryEncoder`:
   - 使用一维卷积处理历史观测序列
   - 支持不同时间步长（10, 20, 50）
   - 将历史信息压缩为固定长度表示

4. **权重初始化** `weight_init()`:
   - 正交初始化线性层权重
   - 零初始化偏置

---

### 6. runner/ 目录

#### 6.1 runner/on_constraint_policy_runner.py
**作用**: 训练流程管理和协调

**主要类**: `OnConstraintPolicyRunner`

**核心功能**:

**初始化**:
- 创建Actor-Critic网络
- 创建NP3O算法实例
- 初始化经验回放缓冲区
- 设置TensorBoard日志记录器

**训练循环** `learn()`:
1. 收集经验（Rollout）
   - 在多个环境中执行策略
   - 收集状态、动作、奖励、成本
   - 存储到回放缓冲区

2. 计算回报
   - 计算奖励回报（GAE）
   - 计算成本回报
   - 计算约束违反项

3. 策略更新
   - 小批量采样
   - 计算损失（surrogate + value + cost + violation + imitation）
   - 反向传播和梯度更新
   - 更新学习率（自适应KL）

4. 日志记录
   - 记录各项损失
   - 记录平均奖励和episode长度
   - 记录FPS和时间统计
   - 保存模型检查点

**其他方法**:
- `save()`: 保存模型和优化器状态
- `load()`: 加载预训练模型
- `get_inference_policy()`: 获取推理策略
- `log()`: 记录训练信息到TensorBoard

**损失函数组成**:
```
Total Loss = 
    surrogate_loss +           # PPO surrogate损失
    cost_viol_loss_coef * viol_loss +  # 约束违反损失
    value_loss_coef * value_loss +      # 价值函数损失
    cost_value_loss_coef * cost_value_loss +  # 成本价值函数损失
    -entropy_coef * entropy +           # 熵正则化
    imi_weight * imitation_loss         # 模仿学习损失（可选）
```

#### 6.2 runner/rollout_storage.py
**作用**: 经验回放缓冲区实现

**主要类**:

**RolloutStorage类**（标准PPO）:
- 存储观测、动作、奖励、价值等
- 计算GAE（Generalized Advantage Estimation）
- 小批量生成器

**RolloutStorageWithCost类**（NP3O）:
- 继承自RolloutStorage
- 额外存储成本相关信息
  - `costs`: 成本值
  - `cost_values`: 成本价值估计
  - `cost_returns`: 成本回报
  - `cost_advantages`: 成本优势
  - `cost_violation`: 约束违反项

**关键方法**:
- `add_transitions()`: 添加转换
- `compute_returns()`: 计算奖励回报和优势
- `compute_cost_returns()`: 计算成本回报和优势
- `mini_batch_generator()`: 生成小批量训练数据
- `reccurent_mini_batch_generator()`: 生成RNN用的小批量数据

**存储格式**:
```
[time_steps, num_envs, ...]
```

---

### 7. utils/ 目录

#### 7.1 utils/task_registry.py
**作用**: 任务注册表，管理环境和配置

**主要类**: `TaskRegistry`

**核心功能**:
- `register()`: 注册任务、环境配置和训练配置
- `make_env()`: 创建环境实例
- `make_alg_runner()`: 创建算法运行器

**使用流程**:
1. 注册任务（在配置文件中）
2. 通过名称创建环境
3. 创建对应的训练运行器

**全局实例**:
```python
task_registry = TaskRegistry()
```

#### 7.2 utils/helpers.py
**作用**: 辅助工具函数

**主要功能**:
- 命令行参数解析
- 配置更新
- 类到字典转换
- 随机种子设置
- 仿真参数解析
- 学习率调度
- 模型加载

#### 7.3 utils/logger.py
**作用**: 日志记录工具

**功能**:
- TensorBoard日志记录
- 控制台输出
- 指标统计

#### 7.4 utils/math.py
**作用**: 数学工具函数

**主要函数**:
- 四元数运算
- 姿态角转换
- 向量运算
- 归一化和缩放

#### 7.5 utils/terrain.py
**作用**: 地形生成和管理

**功能**:
- 生成各种地形（平地、障碍物、阶梯等）
- 地形课程学习
- 地形可视化
- 高度场生成

---

### 8. resources/ 目录

#### 8.1 resources/titati/
**作用**: Titati机器人资源文件

**子目录**:

**urdf/**:
- `titi_description.urdf`: 机器人描述文件
- `wheeled_titatit_rl.urdf`: 轮式机器人描述文件

**dae/**:
- 各部件的Collada模型文件
- 包括：base（基座）、FL/FR/RL/RR（四条腿的battery/hip/thigh/calf/foot）

**meshes/**:
- 各部件的STL网格文件
- 用于物理碰撞和可视化

---

## 训练流程

### 1. 训练启动
```bash
python train.py --task=TitatiRough
```

### 2. 训练步骤

#### 阶段1：环境初始化
1. 加载配置文件
2. 创建Isaac Gym仿真器
3. 生成地形
4. 创建多个并行环境实例

#### 阶段2：网络初始化
1. 创建Actor-Critic网络
2. 初始化NP3O算法
3. 创建经验回放缓冲区

#### 阶段3：训练循环
对于每个训练迭代：

**Rollout阶段**:
1. 在所有环境中执行当前策略
2. 收集状态、动作、奖励、成本
3. 存储到回放缓冲区
4. 更新历史缓冲区

**学习阶段**:
1. 计算奖励回报（GAE）
2. 计算成本回报
3. 计算约束违反项
4. 多个epoch的小批量更新：
   - 计算各种损失
   - 反向传播
   - 梯度裁剪
   - 更新参数
   - 调整学习率（基于KL散度）

**日志和保存**:
1. 记录损失和指标
2. 保存模型检查点
3. 输出训练进度

### 3. 训练技巧

**域随机化**:
- 物理参数随机化（摩擦力、恢复系数）
- 质量和质心随机化
- PD增益随机化
- 电机强度随机化
- 动作延迟模拟

**课程学习**:
- 地形难度递增
- 命令范围扩展
- 最大episode长度调整

**约束处理**:
- 动态调整k_value
- 多个成本函数
- 约束违反惩罚

---

## 测试和部署

### 1. 测试脚本
```bash
python simple_play.py
```

### 2. 模型导出
训练完成后可以导出为ONNX格式用于部署：
```python
model.save_torch_jit_policy(path, device)
```

### 3. Sim2Real
代码中包含sim2real相关的索引映射：
```python
def reindex(self, tensor):
    # 将仿真器关节顺序转换为SDK顺序
    return tensor[:,[4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11]]
```

---

## 关键技术点

### 1. 约束强化学习（CRL）
- 使用NP3O算法处理成本约束
- 支持多个约束条件
- 动态调整约束权重

### 2. 模仿学习
- BarlowTwins自监督学习
- Teacher-Student架构
- 渐进式模仿权重衰减

### 3. 状态估计
- 历史信息编码（CNN）
- 特权信息利用（训练时）
- 传感器融合

### 4. 域随机化
- 广泛的物理参数随机化
- 提高sim2real迁移性能

### 5. 课程学习
- 地形难度递增
- 任务复杂度渐进

---

## 配置说明

### 关键参数

**环境参数**:
- `num_envs`: 并行环境数量
- `episode_length_s`: Episode长度（秒）
- `dt`: 时间步长

**观察空间**:
- `n_proprio`: 本体感知维度
- `n_priv_latent`: 特权信息维度
- `history_len`: 历史长度
- `contact_buf_len`: 接触缓冲长度

**动作空间**:
- `num_actions`: 动作维度（16个关节）
- `action_scale`: 动作缩放因子

**奖励权重**:
- `lin_vel_z`: 线速度跟踪
- `ang_vel_z`: 角速度跟踪
- `orientation`: 姿态稳定性
- `torque_smooth`: 扭矩平滑
- `dof_pos_limits`: 关节位置限制

**成本约束**:
- `joint_pos_err`: 关节位置误差
- `joint_vel_err`: 关节速度误差
- `torque_limits`: 扭矩限制

**训练参数**:
- `learning_rate`: 学习率
- `num_learning_epochs`: 学习epoch数
- `num_mini_batches`: 小批量数
- `clip_param`: PPO clip参数
- `gamma`: 折扣因子
- `lambda`: GAE参数

---

## 日志和可视化

### TensorBoard
训练日志保存在 `logs/` 目录：
```bash
tensorboard --logdir logs/
```

**可查看的指标**:
- 损失曲线（value, cost, surrogate, viol, imitation）
- 奖励统计
- Episode长度
- FPS和训练时间
- 动作噪声标准差

### 模型检查点
模型保存在日志目录下：
- `model_0.pt`: 初始模型
- `model_100.pt`, `model_200.pt`, ...: 定期保存的检查点
- 每个检查点包含：模型权重、优化器状态、迭代次数

---

## 依赖项

**主要依赖**:
- Isaac Gym: 物理仿真
- PyTorch: 深度学习框架
- NumPy: 数值计算
- TensorBoard: 可视化

**Python库**:
- torch
- numpy
- torchvision
- opencv-python
- tensorboard

---

## 扩展和修改

### 添加新的奖励函数
1. 在配置中添加奖励权重
2. 在`LeggedRobot`中实现`_reward_<name>()`方法
3. 在`compute_reward()`中自动调用

### 添加新的成本约束
1. 在配置中添加成本参数
2. 在`LeggedRobot`中实现`_cost_<name>()`方法
3. 在`compute_cost()`中自动调用

### 修改网络结构
1. 修改`ActorCriticBarlowTwins`类
2. 调整配置中的网络维度
3. 确保输入输出维度匹配

### 更换地形
1. 修改`terrain.cfg`中的参数
2. 在`utils/terrain.py`中实现新地形
3. 调整地形课程学习策略

---

## 故障排除

**常见问题**:
1. **训练不稳定**: 调整学习率、增加batch size、检查奖励函数
2. **机器人摔倒**: 调整PD增益、增加稳定性奖励、检查初始姿态
3. **违反约束**: 调整k_value、增加cost权重、检查约束定义
4. **内存不足**: 减少环境数量、减小batch size、减小网络规模

---

## 参考文献

1. **PPO**: Schulman et al., "Proximal Policy Optimization Algorithms", 2017
2. **BarlowTwins**: Zbontar et al., "Barlow Twins: Self-Supervised Learning via Redundancy Reduction", 2021
3. **Constraint RL**: Ray et al., "Benchmarking Safe Exploration in Deep Reinforcement Learning", 2019

---

## 联系方式

- GitHub: https://github.com/DDTRobot/quadruped-wheel-titatit-rl.git
- License: See LICENSE file

---

*文档生成时间: 2026年4月7日*
*项目版本: 基于commit 4ba7d2d*