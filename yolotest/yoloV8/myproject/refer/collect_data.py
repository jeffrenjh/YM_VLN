import os
from Robotic_Arm.rm_robot_interface import *
import numpy as np
import pyrealsense2 as rs
import time
import cv2
from data_dual import CollectData

def initialize_robot(robot_ip, thread_mode=None, connection_level=3):
    """Initialize robot arm controller and establish connection"""
    print(f"\nInitializing robot at {robot_ip} with connection level {connection_level}...")
    
    # Create a new instance with the specified thread mode
    if thread_mode is not None:
        print(f"Using thread mode: {thread_mode}")
        robot_controller = RoboticArm(thread_mode)
    else:
        print("Using default thread mode")
        # Default to single thread mode if none specified
        robot_controller = RoboticArm()
    
    # Try to connect with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        print(f"Connecting to robot at {robot_ip}, attempt {attempt+1}/{max_retries}...")
        handle = robot_controller.rm_create_robot_arm(robot_ip, 8080, connection_level)
        
        if handle.id != -1:
            print(f"Successfully connected to robot at {robot_ip}, handle ID: {handle.id}")
            # Verify connection is active
            succ, state = robot_controller.rm_get_current_arm_state()
            if succ == 0:
                print(f"Connection verified for robot at {robot_ip}")
                print(f"Current state: {state}")
                
                # Get robot info for additional verification
                succ, info = robot_controller.rm_get_robot_info()
                if succ == 0:
                    print(f"Robot info: {info}")
                
                return robot_controller
            else:
                print(f"Connection established but couldn't get state from robot at {robot_ip}. Error code: {succ}")
        else:
            print(f"Failed to create robot arm handle for {robot_ip}. Handle ID: {handle.id}")
        
        if attempt < max_retries - 1:
            print(f"Failed to connect to robot at {robot_ip}, retrying in 2 seconds...")
            time.sleep(2)
    
    print(f"Failed to connect to robot at {robot_ip} after {max_retries} attempts")
    return None


# Define start position (in degrees)
START_POSITION_ANGLE_LEFT_ARM = [
    65,   # Joint 1
    38,    # Joint 2
    -66,  # Joint 3
    12,   # Joint 4
    6,  # Joint 5
    119,    # Joint 6
    66    # Joint 7
]

# Define start position (in degrees)
START_POSITION_ANGLE_RIGHT_ARM = [
    -59,   # Joint 1
    38,    # Joint 2
    -123,  # Joint 3
    -9,   # Joint 4
    -10,  # Joint 5
    -120,    # Joint 6
    32    # Joint 7
]


# Camera serial numbers configuration
CAMERA_SERIALS = {
    'head': '153122070447',  # Replace with actual serial number
    'left_wrist': '427622270438',   # Replace with actual serial number
    'right_wrist': '427622270277',   # Replace with actual serial number
}

def move_to_start_position_with_angles(robot_controller, start_angles, arm_name):
    """Move robot to the specified start position"""
    print(f"\nMoving {arm_name} arm to start position...")
    
    # Check if robot is still connected
    succ, _ = robot_controller.rm_get_current_arm_state()
    if succ != 0:
        print(f"Error: {arm_name} arm is not connected or responding")
        return False
    
    # Get current joint positions
    succ, state = robot_controller.rm_get_current_arm_state()
    if succ == 0:
        current_joints = state['joint']
        print(f"Current {arm_name} arm position: {current_joints}")
    
    # Move to start position with error handling
    try:
        print(f"Target {arm_name} arm position: {start_angles}")
        result = robot_controller.rm_movej(start_angles, 20, 0, 0, 1)  # v=20%, blocking=True
        if result == 0:
            print(f"Successfully moved {arm_name} arm to start position")
            # Verify current position
            succ, state = robot_controller.rm_get_current_arm_state()
            if succ == 0:
                current_joints = state['joint']
                print(f"New {arm_name} arm position: {current_joints}")
                max_diff = max(abs(np.array(current_joints) - np.array(start_angles)))
                if max_diff > 0.01:  # Allow small tolerance of 0.01 radians
                    print(f"Warning: {arm_name} arm position differs from target by {max_diff} radians")
            else:
                print(f"Warning: Could not verify {arm_name} arm position")
            # Wait for system to stabilize
            print(f"Waiting for {arm_name} arm to stabilize...")
            time.sleep(2)
            return True
        else:
            print(f"Failed to move {arm_name} arm to start position. Error code: {result}")
            return False
    except Exception as e:
        print(f"Exception while moving {arm_name} arm: {str(e)}")
        return False

# Keep the original function for backward compatibility
def move_to_start_position(robot_controller):
    """Move robot to the predefined start position (deprecated)"""
    print("Warning: Using deprecated move_to_start_position function")
    print("\nMoving to start position...")
    
    # Determine which start position to use based on the robot's IP
    succ, info = robot_controller.rm_get_robot_info()
    if succ == 0 and 'ip' in info:
        if info['ip'] == "192.168.1.19":
            start_angles = START_POSITION_ANGLE_LEFT_ARM
            arm_name = "left"
        else:
            start_angles = START_POSITION_ANGLE_RIGHT_ARM
            arm_name = "right"
    else:
        # Default to right arm if we can't determine
        start_angles = START_POSITION_ANGLE_RIGHT_ARM
        arm_name = "unknown"
    
    return move_to_start_position_with_angles(robot_controller, start_angles, arm_name)


def initialize_realsense():
    """Initialize RealSense context and check for connected devices"""
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) < 3:
        print(f"Error: Found only {len(devices)} RealSense devices. Need 3 devices.")
        exit(1)
    
    print("\nDetected RealSense devices:")
    for i, dev in enumerate(devices):
        print(f"Device {i}: {dev.get_info(rs.camera_info.name)} (SN: {dev.get_info(rs.camera_info.serial_number)})")
    
    return ctx, devices

def find_device_by_serial(devices, serial):
    """Find device index by serial number"""
    for i, dev in enumerate(devices):
        if dev.get_info(rs.camera_info.serial_number) == serial:
            return i
    return None

def initialize_cameras(devices):
    """Initialize and start streaming for all cameras"""
    pipelines = {}
    configs = {}
    
    for camera_name, serial in CAMERA_SERIALS.items():
        device_idx = find_device_by_serial(devices, serial)
        if device_idx is None:
            print(f"Error: Could not find {camera_name} camera with serial number {serial}")
            exit(1)
            
        pipelines[camera_name] = rs.pipeline()
        configs[camera_name] = rs.config()
        
        # Enable device by serial number
        configs[camera_name].enable_device(serial)
        
        # Enable streams
        configs[camera_name].enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        configs[camera_name].enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        tag = False
        # Start streaming
        try:
            pipelines[camera_name].start(configs[camera_name])
            print(f"Started {camera_name} camera (SN: {serial})")

            # # 硬件重置（GitHub issue #6628 解决方案）
            # device = profile.get_device()
            # print(f"Hardware resetting {camera_name} camera...")
            # # if tag == False:
            # #     tag = True
            # #     device.hardware_reset()
            
            # # 复位后需要等待相机重新枚举
            # time.sleep(2)
            
            # # 再次启动 pipeline
            # pipelines[camera_name].stop()
            # pipelines[camera_name] = rs.pipeline()
            # profile = pipelines[camera_name].start(configs[camera_name])
            # print(f"{camera_name} restarted after reset.")
            
            # # 热身阶段（丢掉前几帧）
            # print(f"Warming up {camera_name} camera...")
            # for _ in range(10):
            #     frames = pipelines[camera_name].wait_for_frames()
            #     time.sleep(0.05)


        except RuntimeError as e:
            print(f"Error starting {camera_name} camera: {str(e)}")
            exit(1)
    # for name, pipe in pipelines.items():
    #     print(f"Warming up camera: {name}")
    #     for _ in range(5):  # 丢掉前5帧
    #         frames = pipe.wait_for_frames()
    #         time.sleep(0.05)  

    return pipelines

def collect(left_robot_controller, right_robot_controller, pipelines, index):
    # Dictionary to store all camera images
    imgs = {}
    
    # Collect frames from all cameras
    for camera_name in pipelines.keys():
        frames = pipelines[camera_name].wait_for_frames()
        
        # Get color frame
        color_frame = frames.get_color_frame()
        if not color_frame:
            print(f"Failed to get color frame from {camera_name} camera")
            return None
            
        # Get depth frame
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            print(f"Failed to get depth frame from {camera_name} camera")
            return None
            
        # Convert frames to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # Store images in dictionary
        imgs[camera_name] = {
            'color': color_image,
            'depth': depth_image
        }
    
    # Get robot state with error handling
    try:
        succ, left_status = left_robot_controller.rm_get_current_arm_state()
        if succ != 0:
            print(f"Error getting left arm state: {succ}")
            return None
        # Uncomment for debugging
        # print(f"Left arm state: {left_status}")

        succ, right_status = right_robot_controller.rm_get_current_arm_state()
        if succ != 0:
            print(f"Error getting right arm state: {succ}")
            return None
        # Uncomment for debugging
        # print(f"Right arm state: {right_status}")
        
        # Get gripper state
        succ, left_gripper_state = left_robot_controller.rm_get_gripper_state()
        if succ != 0:
            print(f"Error getting left gripper state: {succ}")
            return None
        left_gripper = left_gripper_state['actpos']
        left_gripper = (float(left_gripper)) / 1000
        # Uncomment for debugging
        # print(f"Left gripper: {left_gripper}")

        succ, right_gripper_state = right_robot_controller.rm_get_gripper_state()
        if succ != 0:
            print(f"Error getting right gripper state: {succ}")
            return None
        right_gripper = right_gripper_state['actpos']
        right_gripper = (float(right_gripper)) / 1000
        # Uncomment for debugging
        # print(f"Right gripper: {right_gripper}")

        # Uncomment for debugging status dictionaries
        # print(f"Right status keys: {right_status.keys()}")
        # print(f"Right status joint: {right_status['joint']}")
        # print(f"Right status pose: {right_status['pose']}")
        # print(f"Left status keys: {left_status.keys()}")
        # print(f"Left status joint: {left_status['joint']}")
        # print(f"Left status pose: {left_status['pose']}")
        
        # Create and return the data object
        return CollectData(right_status, right_gripper, left_status, left_gripper, imgs)
    except Exception as e:
        print(f"Error in collect function: {str(e)}")
        return None

def create_recording_directory():
    """Create a new directory for the recording session"""
    # Get the workspace root directory (two levels up from the current file)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(os.path.dirname(current_dir))
    
    # Create path for data collection
    data_dir = os.path.join(workspace_root, "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Create a new session directory with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(data_dir, f"session_{timestamp}")
    if not os.path.exists(session_dir):
        os.makedirs(session_dir)
    
    return session_dir

def calculate_movement_delta(current_pos, last_pos):
    """Calculate movement delta between current and last position"""
    # Ensure we're working with NumPy arrays
    current = np.array(current_pos)
    last = np.array(last_pos)
    
    # Calculate the absolute difference for each position component
    delta = np.abs(current[:12] - last[:12])
    return delta

def should_save_frame(delta, left_gripper_change, right_gripper_change):
    """Determine if the current frame should be saved based on movement"""
    movement_threshold = 0.01
    gripper_threshold = 0.1
    
    # Uncomment for debugging
    # print("\nJoint movements:")
    # for i, d in enumerate(delta):
    #     print(f"Joint {i+1}: {d:.4f}")
    # print(f"Left gripper change: {left_gripper_change:.4f}")
    # print(f"Right gripper change: {right_gripper_change:.4f}")
    
    # Handle NumPy arrays correctly
    if isinstance(delta, np.ndarray):
        significant_movement = np.any(delta > movement_threshold)
    else:
        significant_movement = any(d > movement_threshold for d in delta)
        
    significant_gripper = abs(left_gripper_change) >= gripper_threshold or abs(right_gripper_change) >= gripper_threshold
    
    # Uncomment for debugging
    # print(f"Significant movement: {significant_movement}, Significant gripper: {significant_gripper}")
    return significant_movement or significant_gripper

def cleanup(pipelines):
    """Clean up resources before exiting"""
    for pipeline in pipelines.values():
        pipeline.stop()
    cv2.destroyAllWindows()

def disconnect_robot(robot):
    """Disconnect from the robot arm"""
    handle = robot.rm_delete_robot_arm()
    if handle == 0:
        print("Successfully disconnected from the robot arm")
    else:
        print("Failed to disconnect from the robot arm")

def main():
    # Initialize robot arms exactly like the successful example
    # First robot with triple thread mode (mode=2)
    print("Initializing first robot (right arm)...")
    right_robot_controller = initialize_robot("192.168.1.18", rm_thread_mode_e(2), connection_level=3)
    if right_robot_controller is None:
        print("Failed to initialize right robot arm, exiting")
        return
    
    print("Right robot arm initialized successfully")
    
    # Add a delay before initializing the second robot
    print("Waiting for system to stabilize before initializing second robot...")
    
    # Second robot with default thread mode
    print("Initializing second robot (left arm)...")
    left_robot_controller = initialize_robot("192.168.1.19", connection_level=3)
    if left_robot_controller is None:
        print("Failed to initialize left robot arm, exiting")
        # Make sure to disconnect the first robot before exiting
        disconnect_robot(right_robot_controller)
        return
    
    # Verify both robots are still connected
    print("Verifying robot connections...")
    right_succ, _ = right_robot_controller.rm_get_current_arm_state()
    left_succ, _ = left_robot_controller.rm_get_current_arm_state()
    
    if right_succ != 0 or left_succ != 0:
        print("Error: One or both robot arms disconnected")
        # Try to disconnect any connected robots
        if right_succ == 0:
            disconnect_robot(right_robot_controller)
        if left_succ == 0:
            disconnect_robot(left_robot_controller)
        return
    
    print("Both robot arms initialized successfully")
    
    # Initialize RealSense cameras
    print("Initializing RealSense cameras...")
    ctx, devices = initialize_realsense()
    pipelines = initialize_cameras(devices)
    
    # Move to start position
    print("\nMoving to start position...")
    # Fix the START_POSITION_ANGLE reference in move_to_start_position
    # by passing the correct start position for each arm
    if not move_to_start_position_with_angles(left_robot_controller, START_POSITION_ANGLE_LEFT_ARM, "left"):
        print("Failed to move to start left arm position")
        disconnect_robot(left_robot_controller)
        disconnect_robot(right_robot_controller)
        cleanup(pipelines)
        return
    if not move_to_start_position_with_angles(right_robot_controller, START_POSITION_ANGLE_RIGHT_ARM, "right"):
        print("Failed to move to start right arm position")
        disconnect_robot(left_robot_controller)
        disconnect_robot(right_robot_controller)
        cleanup(pipelines)
        return
    
    print("Both arms successfully moved to start positions")
    
    # Initialize recording state
    index = 0
    
    # Register cleanup handlers to ensure robots are properly disconnected
    import atexit
    atexit.register(lambda: disconnect_robot(left_robot_controller))
    atexit.register(lambda: disconnect_robot(right_robot_controller))
    
    last_action = 0
    directory_path = create_recording_directory()
    print(f"\nStarted recording. Saving data to: {directory_path}")
    
    try:
        while True:
            # Execute data collection operation
            data_tmp = collect(left_robot_controller, right_robot_controller, pipelines, index)
            if data_tmp is None:
                print("Failed to collect data, checking robot connections...")
                
                # Verify robot connections
                right_succ, _ = right_robot_controller.rm_get_current_arm_state()
                left_succ, _ = left_robot_controller.rm_get_current_arm_state()
                
                if right_succ != 0 or left_succ != 0:
                    print("Robot connection lost. Exiting...")
                    break
                    
                print("Robot connections are still active. Retrying data collection...")
                time.sleep(1)
                continue

            if last_action == 0:
                index += 1
                print("First frame collected")
                data_tmp.write(directory_path, index)
                # Convert NumPy array to list before appending
                last_action = data_tmp.pos.tolist()
                last_action.append(data_tmp.gripper[1])  # Use first element of gripper array
                last_action.append(data_tmp.gripper[0])  # Use first element of gripper array
                # Uncomment for debugging
                # print(f"Initial last_action: {last_action}")
            else:
                # Uncomment for debugging
                # print("size", len(data_tmp.pos), len(last_action))
                # print("action", last_action)
                # print("data_tmp.pos", data_tmp.pos)
                delta = calculate_movement_delta(data_tmp.pos, last_action[:12])
                left_gripper_change = data_tmp.gripper[0] - last_action[12]  # Use first element of gripper array
                right_gripper_change = data_tmp.gripper[1] - last_action[13]  # Use first element of gripper array
                
                if should_save_frame(delta, left_gripper_change, right_gripper_change):
                    index += 1
                    data_tmp.write(directory_path, index)
                    # Convert NumPy array to list before appending
                    last_action = data_tmp.pos.tolist()
                    last_action.append(data_tmp.gripper[1])  # Use first element of gripper array
                    last_action.append(data_tmp.gripper[0])  # Use first element of gripper array
                    # Uncomment for debugging
                    # print(f"Updated last_action: {last_action}")
                    print(f"Saved frame {index}")
                else:
                    print("No significant movement detected, skipping...")

            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\nRecording stopped by user")
    finally:
        # Clean up resources
        print("Cleaning up resources...")
        cleanup(pipelines)
        
        # Properly disconnect from both robots
        print("Disconnecting from robots...")
        try:
            disconnect_robot(left_robot_controller)
        except Exception as e:
            print(f"Error disconnecting left robot: {str(e)}")
            
        try:
            disconnect_robot(right_robot_controller)
        except Exception as e:
            print(f"Error disconnecting right robot: {str(e)}")
            
        print("Done")

if __name__ == "__main__":
    main()
