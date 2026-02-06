import numpy as np
import time
from scipy.spatial.transform import Rotation as R
from franka_interface import FrankaInterface  # 确保该文件在同一目录下

def run_trajectory(npz_path, robot_ip='192.168.1.111'):
    # 1. 加载数据
    print(f"正在加载数据: {npz_path}")
    data = np.load(npz_path)
    ee_pts = data['ee_pts']       # (N, 3)
    ee_oris = data['ee_oris']     # (N, 3, 3)
    ee_widths = data['ee_widths'] # (N,)

    # 2. 初始化机器人接口
    print(f"正在连接机器人: {robot_ip}")
    robot = FrankaInterface(ip=robot_ip)

    # --- 安全参数配置 ---
    Z_SAFE_MIN = 0.02   # 最小高度 2cm，防止撞桌子
    Z_SAFE_MAX = 0.60   # 最大高度
    GRIPPER_MAX = 0.08  # Franka 夹爪物理极限
    HZ = 15             # 你的 npz 数据采样率
    
    # --- 阻抗控制参数 ---
    # 刚度 Kx: [x, y, z, rx, ry, rz]
    Kx = np.array([600, 600, 600, 30, 30, 30]) 
    Kxd = np.array([50, 50, 50, 2, 2, 2])     # 阻尼
    robot.start_cartesian_impedance(Kx, Kxd)

    try:
        dt = 1.0 / HZ
        print("开始执行轨迹...")

        for i in range(len(ee_pts)):
            loop_start = time.time()
            
            # 旋转矩阵转四元数 [qx, qy, qz, qw]
            quat = R.from_matrix(ee_oris[i]).as_quat()
            
            # 组合成 7D 目标输入
            target_pose = np.concatenate([p, quat])
            
            # --- B. 夹爪处理 ---
            # 方案：将人手宽度 [0.02, 0.15] 映射到机器人 [0.0, 0.08]
            raw_w = ee_widths[i]
            mapped_w = (raw_w - 0.02) * (0.08 / 0.13) 
            target_w = np.clip(mapped_w, 0.0, GRIPPER_MAX)

            # --- C. 发送指令 ---
            robot.update_desired_ee_pose(target_pose)
            
            # 夹爪不需要每帧都发，可以降低频率发送以减轻通信压力
            if i % 3 == 0:
                robot.set_gripper_position(target_w)

            # --- D. 频率控制 (15Hz) ---
            elapsed = time.time() - loop_start
            sleep_time = (1.0 / HZ) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        print("轨迹执行完毕。")

    except Exception as e:
        print(f"运行中发生错误: {e}")
    finally:
        # 5. 清理与停止
        print("正在停止控制器...")
        robot.terminate_current_policy()
        # 建议最后把夹爪张开，方便下次操作
        # robot.set_gripper_position(0.08)

if __name__ == "__main__":
    # 替换成你实际的 npz 文件名
    NPZ_FILE = "./actions_left_single_arm.npz" 
    run_trajectory(NPZ_FILE)