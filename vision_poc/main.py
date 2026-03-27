import cv2
import torch
import os
import numpy as np
from src.detector import ClimbingDetector
from src.processor import PoseProcessor
from src.classifier import ClimbingLSTM

def run_inference_sequence(video_in, video_out, start_sec, end_sec):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Device: {device}")

    # 1. Module Initialization
    detector = ClimbingDetector()
    processor = PoseProcessor(window_size=30)
    classifier = ClimbingLSTM().to(device)
    
    # Load trained weights if they exist
    if os.path.exists("models/action_lstm.pth"):
        classifier.load_state_dict(torch.load("models/action_lstm.pth", map_location=device))
    classifier.eval()

    # 2. Video Handling
    cap = cv2.VideoCapture(video_in)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate boundaries
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    total_to_process = end_frame - start_frame

    # Set video to the designated start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Output Writer Configuration
    os.makedirs(os.path.dirname(video_out), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(video_out, fourcc, fps, (width, height))

    print(f"Analyzing sequence: from {start_sec}s to {end_sec}s ({total_to_process} frames)")

    current_f = start_frame
    while current_f < end_frame:
        ret, frame = cap.read()
        if not ret: break

        # Pose Inference
        kp_norm, results_yolo = detector.get_poses(frame)
        label = "Stable"
        
        if kp_norm is not None:
            # Temporal Processing
            sequence = processor.prepare_sequence(kp_norm)
            
            # If we have gathered enough frames for the sequence window
            if sequence is not None:
                input_tensor = torch.FloatTensor(sequence).to(device)
                with torch.no_grad():
                    output = classifier(input_tensor)
                    prediction = torch.argmax(output, dim=1).item()
                    label = classifier.classes[prediction]
            
            # Visualization: Skeleton + Status Prediction
            annotated_frame = results_yolo[0].plot() 
            cv2.putText(annotated_frame, f"MOTION PREDICTION | {label}", (50, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            out.write(annotated_frame)
        else:
            # Fallback if no person is detected (prevents crash)
            cv2.putText(frame, "NO DETECTION", (50, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            out.write(frame)

        current_f += 1
        if (current_f - start_frame) % 50 == 0:
            percent = ((current_f - start_frame) / total_to_process) * 100
            print(f"Progress: {percent:.1f}% ({current_f - start_frame}/{total_to_process})")

    cap.release()
    out.release()
    print(f"Sequence successfully saved to: {video_out}")

if __name__ == "__main__":
    run_inference_sequence(
        video_in="data/input/janja_video.mp4", 
        video_out="data/output/janja_sequence_analysis.mp4",
        start_sec=318, 
        end_sec=384
    )