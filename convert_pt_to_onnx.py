#!/usr/bin/env python3
"""
将训练好的 PyTorch 模型 (.pt) 转换为 ONNX 格式
用于模型的部署和推理加速

使用方法:
    python convert_pt_to_onnx.py --pt_path <path> --onnx_path <path> --device <cpu/cuda> --gpu_id <id>
    
示例:
    # 使用CPU转换
    python convert_pt_to_onnx.py --pt_path logs/.../model_6000.pt --onnx_path model_6000.onnx --device cpu
    
    # 使用GPU转换（自动检测）
    python convert_pt_to_onnx.py --pt_path logs/.../model_6000.pt --onnx_path model_6000.onnx --device cuda
    
    # 指定使用单个GPU（推荐用于服务器）
    python convert_pt_to_onnx.py --pt_path logs/.../model_6000.pt --onnx_path model_6000.onnx --device cuda --gpu_id 0
    
    # 批量转换（使用指定的GPU）
    for iter in 1000 2000 3000; do
        python convert_pt_to_onnx.py --pt_path logs/.../model_${iter}.pt --onnx_path model_${iter}.onnx --device cuda --gpu_id 1
    done
"""

import torch
import numpy as np
import os
import argparse
from configs.wheeled_titatit_constraint_him import WheeledTitatitConstraintHimRoughCfg, WheeledTitatitConstraintHimRoughCfgPPO
from modules import ActorCriticBarlowTwins
from global_config import ROOT_DIR


def load_model_and_convert_to_onnx(pt_path, onnx_path, device='cpu'):
    """
    加载 .pt 模型并转换为 ONNX 格式
    
    Args:
        pt_path: .pt 模型文件路径
        onnx_path: 输出的 ONNX 文件路径
        device: 使用的设备 ('cpu' 或 'cuda')
    """
    print("=" * 80)
    print("开始模型转换流程")
    print("=" * 80)
    
    # 1. 加载配置
    print("\n[1/5] 加载模型配置...")
    env_cfg = WheeledTitatitConstraintHimRoughCfg()
    train_cfg = WheeledTitatitConstraintHimRoughCfgPPO()
    
    print(f"  - n_proprio: {env_cfg.env.n_proprio}")
    print(f"  - n_scan: {env_cfg.env.n_scan}")
    print(f"  - n_priv_latent: {env_cfg.env.n_priv_latent}")
    print(f"  - history_len: {env_cfg.env.history_len}")
    print(f"  - num_actions: {env_cfg.env.num_observations - env_cfg.env.n_proprio - env_cfg.env.n_scan - env_cfg.env.history_len * env_cfg.env.n_proprio - env_cfg.env.n_priv_latent}")
    
    # 2. 重建模型结构
    print("\n[2/5] 重建模型结构...")
    
    # 注意：num_actions 应该使用配置中的实际值（16），而不是通过公式计算
    # 训练时使用的 actor_dims 与配置文件不同，需要匹配训练时的配置
    num_actions = 16
    
    actor_critic = ActorCriticBarlowTwins(
        num_prop=env_cfg.env.n_proprio,
        num_scan=env_cfg.env.n_scan,
        num_critic_obs=env_cfg.env.num_observations,
        num_priv_latent=env_cfg.env.n_priv_latent,
        num_hist=env_cfg.env.history_len,
        num_actions=num_actions,
        scan_encoder_dims=train_cfg.policy.scan_encoder_dims,
        actor_hidden_dims=[512, 256, 128, 64, 32],  # 匹配训练时的配置，包含7层
        critic_hidden_dims=train_cfg.policy.critic_hidden_dims,
        activation=train_cfg.policy.activation,
        init_noise_std=train_cfg.policy.init_noise_std,
        priv_encoder_dims=train_cfg.policy.priv_encoder_dims,
        num_costs=train_cfg.policy.num_costs,
        teacher_act=train_cfg.policy.teacher_act,
        imi_flag=train_cfg.policy.imi_flag
    )
    
    print("  模型结构创建完成")
    
    # 3. 加载模型权重
    print(f"\n[3/5] 加载模型权重: {pt_path}")
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"模型文件不存在: {pt_path}")
    
    checkpoint = torch.load(pt_path, map_location=device)
    actor_critic.load_state_dict(checkpoint['model_state_dict'])
    actor_critic.to(device)
    actor_critic.eval()
    
    print(f"  模型加载完成，迭代次数: {checkpoint.get('iter', 'unknown')}")
    
    # 4. 导出为 ONNX 格式
    print(f"\n[4/5] 导出为 ONNX 格式...")
    
    # 获取 actor_teacher_backbone 用于推理
    actor_backbone = actor_critic.actor_teacher_backbone
    
    # 创建示例输入
    # 注意：虽然 MlpBarlowTwinsActor 初始化时 num_hist=5，
    # 但在 act_teacher 方法中实际使用的是 self.num_hist（配置中是10）
    # forward 方法会从历史中取第5个时间步之后的数据进行编码
    num_hist_backbone = env_cfg.env.history_len  # 使用配置中的 history_len (10)
    num_actions = 16  # 实际的输出维度
    
    obs_demo_input = torch.randn(1, env_cfg.env.n_proprio).to(device)
    hist_demo_input = torch.randn(1, num_hist_backbone, env_cfg.env.n_proprio).to(device)
    
    print(f"  - 输入1 (obs) shape: {tuple(obs_demo_input.shape)}")
    print(f"  - 输入2 (obs_hist) shape: {tuple(hist_demo_input.shape)}")
    print(f"  - 输出 (actions) shape: (1, {num_actions})")
    
    # 导出 ONNX (固定 batch_size=1，适配 TensorRT)
    # 注意：对于机器狗实时控制，batch_size 永远是 1，不需要动态维度
    torch.onnx.export(
        actor_backbone,
        (obs_demo_input, hist_demo_input),
        onnx_path,
        export_params=True,        # 存储训练好的参数
        opset_version=11,          # ONNX opset 版本
        do_constant_folding=True,  # 是否执行常量折叠优化
        input_names=['obs', 'obs_hist'],
        output_names=['actions']
    )
    
    print(f"  ONNX 模型已保存到: {onnx_path}")
    
    # 5. 验证 ONNX 模型
    print("\n[5/5] 验证 ONNX 模型...")
    
    try:
        import onnx
        import onnxruntime as ort
        
        # 加载并检查 ONNX 模型
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("  ✓ ONNX 模型结构验证通过")
        
        # 使用 ONNX Runtime 进行推理
        ort_session = ort.InferenceSession(onnx_path)
        
        # 准备输入
        ort_inputs = {
            'obs': obs_demo_input.cpu().numpy(),
            'obs_hist': hist_demo_input.cpu().numpy()
        }
        
        # ONNX Runtime 推理
        ort_outputs = ort_session.run(None, ort_inputs)
        
        # PyTorch 推理
        with torch.no_grad():
            torch_outputs = actor_backbone(obs_demo_input, hist_demo_input)
        
        # 比较输出
        torch_output_np = torch_outputs.cpu().numpy()
        onnx_output_np = ort_outputs[0]
        
        max_diff = np.max(np.abs(torch_output_np - onnx_output_np))
        print(f"  ✓ PyTorch 和 ONNX 输出最大差异: {max_diff:.6f}")
        
        if max_diff < 1e-4:
            print("  ✓ ONNX 模型验证成功！")
        else:
            print("  ⚠ 警告：输出差异较大，请检查转换过程")
            
    except ImportError:
        print("  ⚠ 警告：未安装 onnx 或 onnxruntime，跳过验证步骤")
        print("  要进行验证，请运行: pip install onnx onnxruntime")
    
    print("\n" + "=" * 80)
    print("模型转换完成！")
    print("=" * 80)
    print(f"\nPyTorch 模型: {pt_path}")
    print(f"ONNX 模型:   {onnx_path}")
    print("\n使用示例:")
    print("  import onnxruntime as ort")
    print(f"  session = ort.InferenceSession('{os.path.basename(onnx_path)}')")
    print("  outputs = session.run(None, {'obs': obs_np, 'obs_hist': hist_np})")
    print("=" * 80)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='将 PyTorch 模型 (.pt) 转换为 ONNX 格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用CPU转换
  python convert_pt_to_onnx.py --pt_path logs/.../model_6000.pt --onnx_path model_6000.onnx --device cpu
  
  # 使用GPU转换（自动检测）
  python convert_pt_to_onnx.py --pt_path logs/.../model_6000.pt --onnx_path model_6000.onnx --device cuda
  
  # 使用默认路径和GPU
  python convert_pt_to_onnx.py --device cuda
  
  # 批量转换
  for iter in 1000 2000 3000; do
    python convert_pt_to_onnx.py --pt_path logs/.../model_${iter}.pt --onnx_path model_${iter}.onnx --device cuda
  done
        """
    )
    
    parser.add_argument(
        '--pt_path',
        type=str,
        default=os.path.join(ROOT_DIR, 'logs/rough_wheeled_titatit_constraint/Apr07_01-51-37_test_barlowtwins_feetcontact/model_6000.pt'),
        help='输入的 .pt 模型文件路径'
    )
    
    parser.add_argument(
        '--onnx_path',
        type=str,
        default=os.path.join(ROOT_DIR, 'model_6000.onnx'),
        help='输出的 ONNX 模型文件路径'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'auto'],
        default='auto',
        help='使用的设备: cpu, cuda, 或 auto (自动检测)'
    )
    
    parser.add_argument(
        '--gpu_id',
        type=int,
        default=None,
        help='指定使用的GPU ID（仅当device=cuda时有效），例如: --gpu_id 0 表示使用第一块GPU'
    )
    
    return parser.parse_args()


def detect_device(device_arg, gpu_id=None):
    """
    检测并返回要使用的设备
    
    Args:
        device_arg: 用户指定的设备参数 ('cpu', 'cuda', 'auto')
        gpu_id: 指定的GPU ID（仅当使用cuda时有效）
    
    Returns:
        str: 实际使用的设备
    """
    if device_arg == 'cpu':
        return 'cpu'
    elif device_arg == 'cuda':
        if not torch.cuda.is_available():
            print("⚠ 警告: 请求使用 CUDA 但 CUDA 不可用，将使用 CPU")
            return 'cpu'
        
        # 如果指定了GPU ID
        if gpu_id is not None:
            gpu_count = torch.cuda.device_count()
            if gpu_id >= gpu_count:
                print(f"⚠ 警告: 请求使用 GPU {gpu_id}，但只有 {gpu_count} 个GPU可用")
                print(f"ℹ 将使用 GPU 0")
                gpu_id = 0
            return f'cuda:{gpu_id}'
        return 'cuda'
    elif device_arg == 'auto':
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✓ 检测到 {gpu_count} 个 GPU，将使用 CUDA 加速转换")
            
            # 如果指定了GPU ID
            if gpu_id is not None:
                if gpu_id >= gpu_count:
                    print(f"⚠ 警告: 请求使用 GPU {gpu_id}，但只有 {gpu_count} 个GPU可用")
                    print(f"ℹ 将使用 GPU 0")
                    gpu_id = 0
                return f'cuda:{gpu_id}'
            return 'cuda'
        else:
            print("ℹ 未检测到 GPU，将使用 CPU 转换")
            return 'cpu'
    else:
        print(f"⚠ 未知的设备参数: {device_arg}，将使用 CPU")
        return 'cpu'


if __name__ == '__main__':
    # 解析命令行参数
    args = parse_arguments()
    
    # 检测并设置设备
    device = detect_device(args.device, args.gpu_id)
    
    # 显示转换信息
    print("\n" + "=" * 80)
    print("模型转换配置")
    print("=" * 80)
    print(f"输入模型: {args.pt_path}")
    print(f"输出模型: {args.onnx_path}")
    print(f"使用设备: {device.upper()}")
    
    # 如果使用GPU，显示GPU信息
    if device.startswith('cuda'):
        gpu_idx = 0 if device == 'cuda' else int(device.split(':')[1])
        print(f"GPU ID: {gpu_idx}")
        print(f"GPU 名称: {torch.cuda.get_device_name(gpu_idx)}")
        print(f"GPU 显存: {torch.cuda.get_device_properties(gpu_idx).total_memory / 1024**3:.2f} GB")
        print(f"GPU 计算能力: {torch.cuda.get_device_properties(gpu_idx).major}.{torch.cuda.get_device_properties(gpu_idx).minor}")
    print("=" * 80 + "\n")
    
    # 执行转换
    try:
        load_model_and_convert_to_onnx(args.pt_path, args.onnx_path, device=device)
    except Exception as e:
        print(f"\n❌ 转换失败: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
