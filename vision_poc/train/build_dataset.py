import cv2
import torch
import numpy as np
import os
import sys

# 1. Secure absolute paths
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROJECT_ROOT = os.getcwd()
sys.path.append(PROJECT_ROOT)

from src.detector import ClimbingDetector

def build_dataset(video_path, start_sec, end_sec):
    # Dynamic absolute paths
    model_path = os.path.join(PROJECT_ROOT, 'models', 'yolo11m-pose.pt')
    detector = ClimbingDetector(model_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot read video at {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    start_frame = int(start_sec * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    X_data = [] # Stores pose sequences
    y_data = [] # Stores labels
    
    print(f"Creating Dataset (from second {start_sec} to {end_sec})...")
    
    current_f = 0
    total_frames = int((end_sec - start_sec) * fps)
    
    while current_f < total_frames:
        ret, frame = cap.read()
        if not ret: break
        
        kp_norm, results = detector.get_poses(frame)
        
        if kp_norm is not None:
            # Flatten the 17 points (x,y) into 34 values
            pose_vector = kp_norm.flatten()
            
            # --- CONVERT FRAME TO REAL SECONDS ---
            current_sec = start_sec + (current_f / fps)
            
            # --- ANNOTATION LOGIC ---
            if 318 <= current_sec < 335:      # 318s to 335s
                label = 1                     # Climb
            elif 335 <= current_sec < 342:    # First jump
                label = 2                     # Dyno
            elif 342 <= current_sec < 380:    # Rest / Wait
                label = 0                     
            elif 380 <= current_sec < 383:    # Second jump
                label = 2                     # Dyno
            elif 383 <= current_sec < 386:    # End of boulder
                label = 3                     # Fall/Top
            else:                             
                label = 0                     # Default
                
            X_data.append(pose_vector)
            y_data.append(label)
            
        current_f += 1
        if current_f % 100 == 0:
            print(f"Progress: {current_f}/{total_frames} frames processed.")

    cap.release()
    
    # 2. Secure save in the correct directory
    output_dir = os.path.join(PROJECT_ROOT, 'data', 'dataset')
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, 'X_raw.npy'), np.array(X_data))
    np.save(os.path.join(output_dir, 'y_raw.npy'), np.array(y_data))
    print(f"Dataset saved! ({len(X_data)} poses extracted)")

if __name__ == "__main__":
    video_file = os.path.join(PROJECT_ROOT, "data", "input", "janja_video.mp4")
    build_dataset(video_file, start_sec=318, end_sec=386)