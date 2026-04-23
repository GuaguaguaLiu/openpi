import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置字体和主题
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('default')

def load_data():
    """加载预测和仿真数据"""
    data_sim = pickle.load(open("/home/tione/notebook/workspace/rickyyzliu/code/convert/train_obs_109.pkl", "rb"))
    action_predict_sim = pickle.load(open("/home/tione/notebook/workspace/rickyyzliu/code/openpi/action_list.pkl", "rb"))
    
    print(f"仿真数据形状: {data_sim.shape if hasattr(data_sim, 'shape') else type(data_sim)}")
    print(f"预测数据形状: {action_predict_sim.shape if hasattr(action_predict_sim, 'shape') else type(action_predict_sim)}")
    
    # 详细检查数据内容
    print(f"仿真数据类型: {type(data_sim)}")
    if isinstance(data_sim, list):
        print(f"仿真数据长度: {len(data_sim)}")
        if len(data_sim) > 0:
            print(f"仿真数据第一个元素类型: {type(data_sim[0])}")
            if isinstance(data_sim[0], dict):
                print(f"仿真数据第一个元素键: {list(data_sim[0].keys())}")
    
    print(f"预测数据类型: {type(action_predict_sim)}")
    if isinstance(action_predict_sim, list):
        print(f"预测数据长度: {len(action_predict_sim)}")
        if len(action_predict_sim) > 0:
            print(f"预测数据第一个元素类型: {type(action_predict_sim[0])}")
            if hasattr(action_predict_sim[0], 'shape'):
                print(f"预测数据第一个元素形状: {action_predict_sim[0].shape}")
    
    return data_sim, action_predict_sim

def analyze_data(data_sim, action_predict_sim):
    """分析数据格式和内容"""
    print("\n=== 数据格式分析 ===")
    
    # 检查仿真数据
    if isinstance(data_sim, dict):
        print("仿真数据是字典，键:", list(data_sim.keys()))
        if 'actions' in data_sim:
            sim_actions = data_sim['actions']
            print(f"仿真actions形状: {sim_actions.shape}")
        else:
            print("仿真数据中没有找到'actions'键")
            sim_actions = None
    elif hasattr(data_sim, 'shape'):
        print(f"仿真数据是数组，形状: {data_sim.shape}")
        sim_actions = data_sim
    else:
        print(f"仿真数据类型: {type(data_sim)}")
        # 处理列表数据
        if isinstance(data_sim, list):
            # 检查是否是字典列表
            if len(data_sim) > 0 and isinstance(data_sim[0], dict):
                print("检测到字典列表，提取action字段")
                # 提取action字段
                actions_list = []
                for item in data_sim:
                    if 'action' in item:
                        actions_list.append(item['action'])
                    else:
                        print(f"警告：某个元素没有'action'键，键为: {list(item.keys())}")
                        break
                
                if actions_list:
                    sim_actions = np.array(actions_list)
                    print(f"提取的仿真actions形状: {sim_actions.shape}")
                else:
                    sim_actions = None
                    print("无法提取action数据")
            else:
                sim_actions = np.array(data_sim)
                print(f"仿真数据转换为数组，形状: {sim_actions.shape}")
        else:
            sim_actions = data_sim
    
    # 检查预测数据
    if hasattr(action_predict_sim, 'shape'):
        print(f"预测actions形状: {action_predict_sim.shape}")
        pred_actions = action_predict_sim
    else:
        print(f"预测数据类型: {type(action_predict_sim)}")
        # 处理列表数据
        if isinstance(action_predict_sim, list):
            pred_actions = np.array(action_predict_sim)
            print(f"预测数据转换为数组，形状: {pred_actions.shape}")
        else:
            pred_actions = action_predict_sim
    
    return sim_actions, pred_actions

def plot_joint_angle_comparison(sim_actions, pred_actions, joint_idx=9, max_steps=20, save_path=None):
    """绘制特定关节角度的对比图，检查回溯现象"""
    if sim_actions is None or pred_actions is None:
        print("数据为空，无法绘制")
        return
    
    # 确保数据是numpy数组
    if not isinstance(sim_actions, np.ndarray):
        sim_actions = np.array(sim_actions)
    if not isinstance(pred_actions, np.ndarray):
        pred_actions = np.array(pred_actions)
    
    print(f"仿真数据形状: {sim_actions.shape}")
    print(f"预测数据形状: {pred_actions.shape}")
    
    # 限制数据长度到前20个时间步
    max_steps = min(max_steps, sim_actions.shape[0])
    sim_actions = sim_actions[:max_steps]
    pred_actions = pred_actions[:max_steps]
    
    print(f"限制到前{max_steps}个时间步")
    
    # 处理预测数据的3D结构 (time_steps, action_horizon, action_dim)
    if pred_actions.ndim == 3:
        print("处理3D预测数据，提取每个时间步的10个动作序列")
        time_steps, action_horizon, action_dim = pred_actions.shape
        
        # 提取特定关节角度
        pred_joint = pred_actions[:, :, joint_idx]  # (time_steps, action_horizon)
        
        print(f"预测关节角度形状: {pred_joint.shape}")
    else:
        print("预测数据不是3D，直接提取关节角度")
        pred_joint = pred_actions[:, joint_idx]
    
    # 提取仿真数据的对应关节角度
    if sim_actions.ndim == 2:
        sim_joint = sim_actions[:, joint_idx]
    else:
        print("仿真数据维度不正确")
        return
    
    # 创建对比图
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # 绘制仿真数据（GT）- 每个时间步一个点
    gt_time = np.arange(max_steps)
    ax.plot(gt_time, sim_joint, 'bo-', label='Ground Truth', linewidth=3, markersize=6, alpha=0.8)
    
    # 绘制预测数据 - 每个时间步预测未来10步
    if pred_actions.ndim == 3:
        for t in range(max_steps):
            # 每个时间步的10个预测值，对应未来t+1到t+10步
            pred_t = pred_joint[t, :]  # 10个预测值
            future_time = np.arange(t+1, t+11)  # 未来10步的时间轴
            ax.plot(future_time, pred_t, 'r--', linewidth=1.5, alpha=0.6)
    else:
        ax.plot(gt_time, pred_joint, 'r--', label='Predicted', linewidth=2, alpha=0.7)
    
    # 计算并显示统计信息
    if pred_actions.ndim == 3:
        pred_mean = np.mean(pred_joint, axis=1)
        mse = np.mean((sim_joint - pred_mean) ** 2)
        mae = np.mean(np.abs(sim_joint - pred_mean))
        corr = np.corrcoef(sim_joint, pred_mean)[0, 1]
    else:
        mse = np.mean((sim_joint - pred_joint) ** 2)
        mae = np.mean(np.abs(sim_joint - pred_joint))
        corr = np.corrcoef(sim_joint, pred_joint)[0, 1]
    
    # 显示统计信息
    ax.text(0.02, 0.98, f'Joint {joint_idx} (Right Arm)\nMSE: {mse:.4f}\nMAE: {mae:.4f}\nCorrelation: {corr:.4f}', 
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 设置坐标轴
    ax.set_title(f'Joint Angle {joint_idx} Comparison - Check for Backtracking (First {max_steps} Steps)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Time Steps', fontsize=14)
    ax.set_ylabel('Joint Angle Value', fontsize=14)
    
    # 设置x轴刻度，扩展到显示预测的未来步数
    max_x = max_steps + 10  # 显示到未来10步
    ax.set_xticks(range(0, max_x, 2))
    ax.set_xlim(-0.5, max_x - 0.5)
    
    # 设置y轴刻度，显示具体数值
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    y_ticks = np.linspace(y_min, y_max, 8)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{y:.3f}' for y in y_ticks], fontsize=10)
    
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 添加说明文字
    ax.text(0.02, 0.02, f'Blue dots: GT (current step)\nRed dashed: Predicted (future 10 steps)', 
            transform=ax.transAxes, fontsize=10, alpha=0.8,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"图片已保存到: {save_path}")
    
    plt.show()

def plot_error_analysis(sim_actions, pred_actions, save_path=None):
    """绘制误差分析图"""
    if sim_actions is None or pred_actions is None:
        return
    
    # 确保数据是numpy数组
    if not isinstance(sim_actions, np.ndarray):
        sim_actions = np.array(sim_actions)
    if not isinstance(pred_actions, np.ndarray):
        pred_actions = np.array(pred_actions)
    
    print(f"误差分析 - 仿真数据形状: {sim_actions.shape}")
    print(f"误差分析 - 预测数据形状: {pred_actions.shape}")
    
    # 处理维度不匹配的情况
    if sim_actions.ndim == 1 and pred_actions.ndim == 3:
        print("误差分析：处理维度不匹配")
        if pred_actions.shape[0] == sim_actions.shape[0]:
            pred_actions = pred_actions[:, 0, :]  # 取第一个动作
            print(f"处理后预测数据形状: {pred_actions.shape}")
        else:
            print("无法匹配时间步数，跳过误差分析")
            return
    
    # 处理维度
    if sim_actions.ndim == 3:
        sim_actions = sim_actions[0]
    if pred_actions.ndim == 3:
        pred_actions = pred_actions[0]
    
    min_steps = min(sim_actions.shape[0], pred_actions.shape[0])
    sim_actions = sim_actions[:min_steps]
    pred_actions = pred_actions[:min_steps]
    
    print(f"误差分析数据: 时间步数={min_steps}, 动作维度={sim_actions.shape[1] if len(sim_actions.shape) > 1 else 1}")
    
    # 计算误差
    if len(sim_actions.shape) == 1 and len(pred_actions.shape) == 1:
        # 都是1D数据
        errors = sim_actions - pred_actions
    elif len(sim_actions.shape) == 1 and len(pred_actions.shape) == 2:
        # 仿真1D，预测2D，需要扩展仿真数据
        errors = sim_actions[:, np.newaxis] - pred_actions
    else:
        # 都是2D数据
        errors = sim_actions - pred_actions
    
    abs_errors = np.abs(errors)
    
    # 创建误差分析图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 误差分布直方图
    axes[0, 0].hist(errors.flatten(), bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[0, 0].set_title('误差分布')
    axes[0, 0].set_xlabel('误差值')
    axes[0, 0].set_ylabel('频次')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 绝对误差随时间变化
    time_steps = np.arange(min_steps)
    mean_abs_error = np.mean(abs_errors, axis=1)
    axes[0, 1].plot(time_steps, mean_abs_error, 'g-', linewidth=2)
    axes[0, 1].set_title('平均绝对误差随时间变化')
    axes[0, 1].set_xlabel('时间步')
    axes[0, 1].set_ylabel('平均绝对误差')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 各维度误差热图
    im = axes[1, 0].imshow(abs_errors.T, aspect='auto', cmap='hot', interpolation='nearest')
    axes[1, 0].set_title('各维度绝对误差热图')
    axes[1, 0].set_xlabel('时间步')
    axes[1, 0].set_ylabel('动作维度')
    plt.colorbar(im, ax=axes[1, 0])
    
    # 4. 误差统计
    axes[1, 1].text(0.1, 0.8, f'总MSE: {np.mean(errors**2):.6f}', transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].text(0.1, 0.7, f'总MAE: {np.mean(abs_errors):.6f}', transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].text(0.1, 0.6, f'最大误差: {np.max(abs_errors):.6f}', transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].text(0.1, 0.5, f'最小误差: {np.min(abs_errors):.6f}', transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].text(0.1, 0.4, f'标准差: {np.std(errors):.6f}', transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].set_title('误差统计')
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
        print(f"误差分析图已保存到: {save_path}")
    
    plt.show()

def main():
    """主函数"""
    print("=== 动作对比分析 ===")
    
    # 加载数据
    data_sim, action_predict_sim = load_data()
    
    # 分析数据格式
    sim_actions, pred_actions = analyze_data(data_sim, action_predict_sim)
    
    # 绘制关节角度对比图
    if sim_actions is not None and pred_actions is not None:
        print("\n=== 绘制关节角度对比图 (检查回溯现象) ===")
        plot_joint_angle_comparison(sim_actions, pred_actions, joint_idx=9, max_steps=20,
                                   save_path="/home/tione/notebook/workspace/rickyyzliu/code/openpi/joint_angle_comparison.png")
    else:
        print("无法进行对比分析，数据格式不支持")

if __name__ == "__main__":
    main()