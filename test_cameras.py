"""Test script for validating camera setup.

This is a wrapper around camera_utils.py that provides additional
testing functionality.

Usage:
    python test_cameras.py
    python test_cameras.py --external-camera 123456789 --wrist-camera 987654321
"""

import sys
import time
import numpy as np
import cv2
import tyro

from camera_utils import list_realsense_devices, RealSenseCamera, MockCamera


def test_cameras(
    external_camera: str = None,
    wrist_camera: str = None,
    use_mock: bool = False,
    duration: int = 10,
):
    """
    Test camera setup and display live feeds.
    
    Args:
        external_camera: Serial number of external camera
        wrist_camera: Serial number of wrist camera
        use_mock: Use mock cameras for testing
        duration: How long to display feeds (seconds)
    """
    print("=" * 70)
    print("Camera Test")
    print("=" * 70)
    
    if use_mock:
        print("\nUsing MOCK cameras (no real hardware)")
    else:
        print("\nListing available RealSense cameras...")
        devices = list_realsense_devices()
        
        if not devices:
            print("\n✗ No RealSense cameras found!")
            print("\nTroubleshooting:")
            print("  1. Check USB 3.0 connections (blue ports)")
            print("  2. Run: rs-enumerate-devices")
            print("  3. Update firmware: realsense-viewer")
            print("  4. Check permissions: sudo usermod -a -G video $USER")
            print("  5. Try: pip install --upgrade pyrealsense2")
            return False
        
        print(f"\nFound {len(devices)} camera(s):")
        for i, dev in enumerate(devices):
            print(f"  [{i}] {dev['name']}")
            print(f"      Serial: {dev['serial_number']}")
            print(f"      Firmware: {dev['firmware_version']}")
        
        if external_camera is None or wrist_camera is None:
            print("\n⚠️  Camera serial numbers not specified!")
            print("Please provide --external-camera and --wrist-camera arguments")
            print("\nExample:")
            if len(devices) >= 2:
                print(f"  python test_cameras.py \\")
                print(f"    --external-camera {devices[0]['serial_number']} \\")
                print(f"    --wrist-camera {devices[1]['serial_number']}")
            else:
                print(f"  python test_cameras.py \\")
                print(f"    --external-camera YOUR_EXTERNAL_SERIAL \\")
                print(f"    --wrist-camera YOUR_WRIST_SERIAL")
            return False
    
    print("\n" + "=" * 70)
    print("Initializing Cameras")
    print("=" * 70)
    
    try:
        if use_mock:
            external_cam = MockCamera(width=640, height=480)
            wrist_cam = MockCamera(width=640, height=480)
        else:
            print(f"\nInitializing external camera (serial: {external_camera})...")
            external_cam = RealSenseCamera(
                serial_number=external_camera,
                width=640,
                height=480,
                fps=30,
            )
            
            print(f"Initializing wrist camera (serial: {wrist_camera})...")
            wrist_cam = RealSenseCamera(
                serial_number=wrist_camera,
                width=640,
                height=480,
                fps=30,
            )
        
        print("\n✓ Cameras initialized successfully!")
        
    except Exception as e:
        print(f"\n✗ Camera initialization failed: {e}")
        return False
    
    print("\n" + "=" * 70)
    print(f"Displaying Camera Feeds ({duration}s)")
    print("=" * 70)
    print("\nPress 'q' to quit early, 's' to save snapshot")
    
    try:
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < duration:
            # Read frames
            ret_ext, ext_img, _ = external_cam.read()
            ret_wrist, wrist_img, _ = wrist_cam.read()
            
            if not ret_ext or not ret_wrist:
                print("\n✗ Failed to capture frames!")
                break
            
            frame_count += 1
            
            # Display side by side
            combined = np.hstack([ext_img, wrist_img])
            
            # Add labels
            cv2.putText(combined, "External Camera", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined, "Wrist Camera", (650, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add FPS counter
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(combined, f"FPS: {fps:.1f}", (10, 460),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Camera Test', combined)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting early...")
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"camera_snapshot_{timestamp}.jpg"
                cv2.imwrite(filename, combined)
                print(f"\nSaved snapshot: {filename}")
        
        print(f"\n✓ Camera test completed!")
        print(f"  Total frames: {frame_count}")
        print(f"  Average FPS: {frame_count / duration:.1f}")
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    
    except Exception as e:
        print(f"\n✗ Error during camera test: {e}")
        return False
    
    finally:
        external_cam.release()
        wrist_cam.release()
        cv2.destroyAllWindows()
    
    print("\n" + "=" * 70)
    print("All Tests Passed! ✓")
    print("=" * 70)
    print("\nYour cameras are working correctly.")
    print("Next steps:")
    print("  1. Update camera serials in config.py")
    print("  2. Test robot: python test_robot.py")
    print("  3. Test policy: python test_policy.py")
    print("  4. Run full system: python main.py --instruction 'your task'")
    
    return True


if __name__ == "__main__":
    success = tyro.cli(test_cameras)
    sys.exit(0 if success else 1)
