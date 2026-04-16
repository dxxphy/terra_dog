# Simple Play 使用示例

## 如何指定特定的模型和log查看视频效果

现在 `simple_play.py` 支持 `--checkpoint_path` 参数，可以灵活加载任何训练checkpoint。

## 基本用法

### 加载默认模型
```bash
python simple_play.py --task=wheeled_titatit
```
这会加载 `model_6000.pt`（默认模型）

### 加载指定路径的模型
```bash
python simple_play.py --task=wheeled_titatit --checkpoint_path logs/rough_wheeled_titatit_constraint/Apr07_01-51-37_test_barlowtwins_feetcontact/model_3000.pt
```

### 加载根目录的模型
```bash
python simple_play.py --task=wheeled_titatit --checkpoint_path model_6000.pt
```

## 实际使用案例

### 案例1：对比不同训练阶段的效果

**训练早期（1000步）：**
```bash
python simple_play.py --task=wheeled_titatit --checkpoint_path logs/rough_wheeled_titatit_constraint/Apr07_01-51-37_test_barlowtwins_feetcontact/model_1000.pt
```

**训练中期（3000步）：**
```bash
python simple_play.py --task=wheeled_titatit --checkpoint_path logs/rough_wheeled_titatit_constraint/Apr07_01-51-37_test_barlowtwins_feetcontact/model_3000.pt
```

**训练后期（6000步）：**
```bash
python simple_play.py --task=wheeled_titatit --checkpoint_path logs/rough_wheeled_titatit_constraint/Apr07_01-51-37_test_barlowtwins_feetcontact/model_6000.pt
```

### 案例2：对比不同训练运行

**第一个训练（Apr07_01-51-37）：**
```bash
python simple_play.py --task=wheeled_titatit --checkpoint_path logs/rough_wheeled_titatit_constraint/Apr07_01-51-37_test_barlowtwins_feetcontact/model_3000.pt
```

**第二个训练（Apr07_15-27-04）：**
```bash
python simple_play.py --task=wheeled_titatit --checkpoint_path logs/rough_wheeled_titatit_constraint/Apr07_15-27-04_test_barlowtwins_feetcontact/model_3000.pt
```

### 案例3：使用绝对路径
```bash
python simple_play.py --task=wheeled_titatit --checkpoint_path /home/dxx/quadruped-wheel-titatit-rl/logs/rough_wheeled_titatit_constraint/Apr07_01-51-37_test_barlowtwins_feetcontact/model_4000.pt
```

## 工作流程建议

### 1. 使用 TensorBoard 查看训练曲线
```bash
tensorboard --logdir logs/rough_wheeled_titatit_constraint
```
在浏览器中打开 `http://localhost:6006` 查看训练曲线，找出性能最好的checkpoint。

### 2. 加载性能最好的模型
假设发现 3500 步的奖励最高：
```bash
python simple_play.py --task=wheeled_titatit --checkpoint_path logs/rough_wheeled_titatit_constraint/Apr07_01-51-37_test_barlowtwins_feetcontact/model_3500.pt
```

### 3. 录制视频并保存
运行后会在当前目录生成 `record.mp4` 文件。

## 注意事项

1. **路径格式**：
   - 相对路径（相对于项目根目录）：`logs/rough_wheeled_titatit_constraint/...`
   - 绝对路径：`/home/dxx/quadruped-wheel-titatit-rl/logs/...`

2. **模型文件**：
   - 必须是 `.pt` 格式的PyTorch模型文件
   - 必须包含 `model_state_dict` 键

3. **视频编码**：
   - 当前使用 MP4V 编码器（可能在某些播放器需要插件）
   - 如需更通用的格式，修改 `simple_play.py` 第120行：
     ```python
     # 改为 MJPEG（最兼容）
     video = cv2.VideoWriter('record.avi', cv2.VideoWriter_fourcc(*'MJPG'), int(1 / env.dt), (img.shape[1],img.shape[0]))
     ```

## 参数说明

- `--task`: 任务名称（默认：wheeled_titatit）
- `--checkpoint_path`: 模型文件的完整路径（新增）
- `--headless`: 无头模式（不显示图形界面）

## 常见问题

**Q: 如何知道有哪些模型可以加载？**
```bash
ls -lh logs/rough_wheeled_titatit_constraint/
ls -lh logs/rough_wheeled_titatit_constraint/Apr07_01-51-37_test_barlowtwins_feetcontact/
```

**Q: 如何批量生成多个checkpoint的视频？**
创建一个脚本：
```bash
#!/bin/bash
for i in 1000 2000 3000 4000 5000 6000; do
    python simple_play.py --task=wheeled_titatit \
        --checkpoint_path logs/rough_wheeled_titatit_constraint/Apr07_01-51-37_test_barlowtwins_feetcontact/model_${i}.pt
    mv record.mp4 record_model_${i}.mp4
done
```

**Q: 如何查看生成的统计信息？**
运行后会在终端输出：
- `action rate`: 动作变化率（越小越平滑）
- `z vel`: Z轴速度（垂直运动，越小越稳定）
- `xy_vel`: XY角速度（旋转运动，越小越稳定）
- `feet air reward`: 脚在空中的奖励

这些指标越低，说明机器人运动越平稳。