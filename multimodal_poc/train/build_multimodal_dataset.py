import cv2
import numpy as np
import os
import sys

# Absolute paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from src.detector import ClimbingDetector
from src.processor import PoseProcessor
from src.audio_engine import ClimbingAudioEngine

def build_multimodal_dataset(video_path, audio_path, start_sec, end_sec):
    print("Initializing Multimodal engines...")
    
    # 1. Initialize "Eyes"
    model_path = os.path.join(PROJECT_ROOT, '..', 'climbing-vision-poc', 'models', 'yolo11m-pose.pt')
    detector = ClimbingDetector(model_path)
    processor = PoseProcessor(window_size=30)
    
    # 2. Initialize "Ears"
    audio_engine = ClimbingAudioEngine(audio_path, fps=30)
    
    # 3. Load Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    X_vision_data, X_audio_data, y_data = [], [], []
    
    print("Synchronized Extraction (Vision + Audio) in progress...")
    
    current_f = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # --- A. VISION BRANCH ---
        kp_norm, _ = detector.get_poses(frame)
        
        if kp_norm is not None:
            # Processor stores the last 30 frames in memory
            seq_vision = processor.prepare_sequence(kp_norm)
            
            # Once we accumulate 30 frames (1 second), we can extract audio
            if seq_vision is not None:
                
                # --- B. AUDIO BRANCH ---
                # Fetch corresponding 30 frames of audio data
                seq_audio = audio_engine.get_audio_window(current_f, window_size=30)
                
                # --- C. ANNOTATION (Real-time offset matching) ---
                current_sec = start_sec + (current_f / fps)
                
                if 318 <= current_sec < 335:      label = 1 # Climb
                elif 335 <= current_sec < 342:    label = 2 # Dyno
                elif 342 <= current_sec < 380:    label = 0 # Rest
                elif 380 <= current_sec < 383:    label = 2 # Dyno
                elif 383 <= current_sec < 386:    label = 3 # Top / Crowd Cheers!
                else:                             label = 0
                
                # --- D. SYNCHRONIZED SAVE ---
                # Store seq_vision[0] because prepare_sequence returns (1, 30, 34)
                X_vision_data.append(seq_vision[0])
                X_audio_data.append(seq_audio)
                y_data.append(label)
                
        current_f += 1
        if current_f % 100 == 0:
            print(f"Progress: {current_f}/{total_frames} synchronized frames.")

    cap.release()
    
    # --- E. SAVE TO DISK ---
    dataset_dir = os.path.join(PROJECT_ROOT, 'data', 'dataset')
    os.makedirs(dataset_dir, exist_ok=True)
    
    np.save(os.path.join(dataset_dir, 'X_vision.npy'), np.array(X_vision_data))
    np.save(os.path.join(dataset_dir, 'X_audio.npy'), np.array(X_audio_data))
    np.save(os.path.join(dataset_dir, 'y.npy'), np.array(y_data))
    
    print("Multimodal Dataset saved!")
    print(f"Vision Shape: {np.array(X_vision_data).shape}")
    print(f"Audio Shape: {np.array(X_audio_data).shape}")
    print(f"Labels Shape: {np.array(y_data).shape}")

if __name__ == "__main__":
    video_in = os.path.join(PROJECT_ROOT, "data", "input", "janja_sequence_318_385.mp4")
    audio_in = os.path.join(PROJECT_ROOT, "data", "input", "janja_audio_track.wav")
    build_multimodal_dataset(video_in, audio_in, start_sec=318, end_sec=386)