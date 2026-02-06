"""Test script for validating Franka robot connection.

This script tests the connection to your Franka robot via ZeroRPC
without loading the policy or cameras.

Usage:
    python test_robot.py
    python test_robot.py --nuc-ip 192.168.1.143 --nuc-port 4242
    python test_robot.py --use-mock  # Test with mock robot
"""

import sys
import time
import numpy as np
import tyro

from franka_interface import FrankaInterface, MockRobot


def test_robot_connection(
    nuc_ip: str = "192.168.1.143",
    nuc_port: int = 4242,
    use_mock: bool = False,
):
    """
    Test connection to Franka robot.
    
    Args:
        nuc_ip: IP address of NUC running Polymetis
        nuc_port: Port of ZeroRPC server
        use_mock: Use mock robot for testing
    """
    print("=" * 70)
    print("Franka Robot Connection Test")
    print("=" * 70)
    print(f"\nNUC IP: {nuc_ip}")
    print(f"NUC Port: {nuc_port}")
    print(f"Mock mode: {use_mock}")
    
    print("\n" + "=" * 70)
    print("Step 1: Connecting to Robot")
    print("=" * 70)
    
    try:
        if use_mock:
            robot = MockRobot(ip=nuc_ip, port=nuc_port)
        else:
            print(f"\nAttempting to connect to {nuc_ip}:{nuc_port}...")
            print("(This may take a few seconds...)")
            robot = FrankaInterface(ip=nuc_ip, port=nuc_port)
        
        print("\n✓ Connected successfully!")
        
    except Exception as e:
        print(f"\n✗ Connection failed: {e}")
        print("\nTroubleshooting:")
        print(f"  1. Verify NUC is accessible: ping {nuc_ip}")
        print(f"  2. Check ZeroRPC server is running on NUC")
        print(f"  3. Verify port {nuc_port} is open")
        print(f"  4. Check network configuration")
        return False
    
    print("\n" + "=" * 70)
    print("Step 2: Reading Robot State")
    print("=" * 70)
    
    try:
        # Get joint positions
        joint_pos = robot.get_joint_positions()
        print(f"\nJoint positions (7D): {joint_pos}")
        
        # Get joint velocities
        joint_vel = robot.get_joint_velocities()
        print(f"Joint velocities (7D): {joint_vel}")
        
        # Get gripper position
        gripper_pos = robot.get_gripper_position()
        print(f"Gripper position (1D): {gripper_pos}")
        
        # Get end-effector pose
        ee_pose = robot.get_ee_pose()
        print(f"End-effector pose (6D): {ee_pose}")
        
        print("\n✓ Successfully read robot state!")
        
    except Exception as e:
        print(f"\n✗ Failed to read robot state: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("Step 3: Testing Joint Impedance Controller")
    print("=" * 70)
    
    try:
        print("\nStarting joint impedance controller...")
        robot.start_joint_impedance(Kq=None, Kqd=None)
        print("✓ Joint impedance controller started!")
        
        time.sleep(0.5)
        
    except Exception as e:
        print(f"\n✗ Failed to start controller: {e}")
        return False
    
    if not use_mock:
        print("\n" + "=" * 70)
        print("Step 4: Testing Small Motion (OPTIONAL)")
        print("=" * 70)
        
        response = input("\nDo you want to test a small robot motion? (y/N): ")
        
        if response.lower() == 'y':
            print("\n⚠️  WARNING: The robot will make a small motion!")
            print("Make sure the workspace is clear and you can reach the E-stop.")
            confirm = input("Type 'YES' to proceed: ")
            
            if confirm == 'YES':
                try:
                    print("\nExecuting small motion test...")
                    
                    # Get current position
                    current_pos = robot.get_joint_positions()
                    print(f"Current position: {current_pos}")
                    
                    # Create small offset (0.05 radians on joint 1)
                    target_pos = current_pos.copy()
                    target_pos[0] += 0.05
                    
                    print(f"Target position: {target_pos}")
                    print("Moving robot (this will take ~2 seconds)...")
                    
                    # Move for 2 seconds
                    start_time = time.time()
                    while time.time() - start_time < 2.0:
                        robot.update_desired_joint_pos(target_pos)
                        time.sleep(0.02)  # 50 Hz
                    
                    print("✓ Motion test completed!")
                    
                    # Return to original position
                    print("Returning to original position...")
                    start_time = time.time()
                    while time.time() - start_time < 2.0:
                        robot.update_desired_joint_pos(current_pos)
                        time.sleep(0.02)
                    
                    print("✓ Returned to original position!")
                    
                except Exception as e:
                    print(f"\n✗ Motion test failed: {e}")
                    print("The robot may be in an error state. Please check the Franka Desk.")
                    return False
            else:
                print("Motion test skipped.")
        else:
            print("Motion test skipped.")
    
    print("\n" + "=" * 70)
    print("Step 5: Cleanup")
    print("=" * 70)
    
    try:
        robot.terminate_current_policy()
        robot.close()
        print("\n✓ Disconnected from robot!")
        
    except Exception as e:
        print(f"\n✗ Cleanup failed: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("All Tests Passed! ✓")
    print("=" * 70)
    print("\nYour robot connection is working correctly.")
    print("Next steps:")
    print("  1. Test cameras: python camera_utils.py")
    print("  2. Test policy: python test_policy.py")
    print("  3. Run full system: python main.py --instruction 'your task'")
    
    return True


if __name__ == "__main__":
    success = tyro.cli(test_robot_connection)
    sys.exit(0 if success else 1)
