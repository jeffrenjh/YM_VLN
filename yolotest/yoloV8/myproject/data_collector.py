import os
import numpy as np
import pyrealsense2 as rs
import time
import cv2
import argparse

def initialize_realsense():
    """Initialize RealSense camera and start streaming"""
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable streams
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    try:
        pipeline.start(config)
        print("RealSense camera initialized successfully")
        
        # Warm up camera (skip first few frames)
        print("Warming up camera...")
        for _ in range(10):
            pipeline.wait_for_frames()
            time.sleep(0.05)
        
        return pipeline
    except RuntimeError as e:
        print(f"Error starting camera: {str(e)}")
        exit(1)

def capture_frame(pipeline):
    """Capture a single frame from the camera"""
    frames = pipeline.wait_for_frames()
    
    # Get color frame
    color_frame = frames.get_color_frame()
    if not color_frame:
        print("Failed to get color frame")
        return None, None
        
    # Get depth frame
    depth_frame = frames.get_depth_frame()
    if not depth_frame:
        print("Failed to get depth frame")
        return None, None
        
    # Convert frames to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    
    return color_image, depth_image

def save_frame(save_dir, index, color_image, depth_image):
    """Save color and depth images to disk"""
    # Save color image
    color_path = os.path.join(save_dir, f"color_{index:06d}.png")
    cv2.imwrite(color_path, color_image)
    
    # Save depth image
    depth_path = os.path.join(save_dir, f"depth_{index:06d}.png")
    cv2.imwrite(depth_path, depth_image)
    
    print(f"Saved frame {index} to {save_dir}")

def create_save_directory(directory_path):
    """Create save directory if it doesn't exist"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")
    else:
        print(f"Using existing directory: {directory_path}")
    
    return directory_path

def cleanup(pipeline):
    """Clean up resources before exiting"""
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Resources cleaned up")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Collect RealSense camera data')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory path to save captured data')
    args = parser.parse_args()
    
    # Create save directory
    save_dir = create_save_directory(args.save_dir)
    
    # Initialize RealSense camera
    print("Initializing RealSense camera...")
    pipeline = initialize_realsense()
    
    # Initialize frame counter
    frame_index = 0
    
    print(f"\nStarted recording. Saving data to: {save_dir}")
    print("Press Ctrl+C to stop recording\n")
    
    try:
        while True:
            # Capture frame
            color_image, depth_image = capture_frame(pipeline)
            
            if color_image is None or depth_image is None:
                print("Failed to capture frame, retrying...")
                time.sleep(0.1)
                continue
            
            # Save frame
            frame_index += 1
            save_frame(save_dir, frame_index, color_image, depth_image)
            
            # Optional: Display preview (uncomment if needed)
            # cv2.imshow('Color', color_image)
            # cv2.waitKey(1)
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\nRecording stopped by user")
    finally:
        # Clean up resources
        print("Cleaning up...")
        cleanup(pipeline)
        print(f"Total frames saved: {frame_index}")
        print("Done")

if __name__ == "__main__":
    main()
