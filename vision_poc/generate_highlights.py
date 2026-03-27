import cv2
import torch
import os
import sys

# Ensure absolute paths for execution from any directory
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

from src.detector import ClimbingDetector
from src.processor import PoseProcessor
from src.classifier import ClimbingLSTM

def create_highlights_from_clip(video_in, video_out):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating Highlights on: {device}")

    # 1. Initialization
    model_path = os.path.join(PROJECT_ROOT, 'models', 'yolo11m-pose.pt')
    lstm_path = os.path.join(PROJECT_ROOT, 'models', 'action_lstm.pth')
    
    detector = ClimbingDetector(model_path)
    processor = PoseProcessor(window_size=30)
    classifier = ClimbingLSTM().to(device)
    
    classifier.load_state_dict(torch.load(lstm_path, map_location=device))
    classifier.eval()

    # 2. Video Configuration (Read from frame 0 to the end)
    cap = cv2.VideoCapture(video_in)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(video_out), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(video_out, fourcc, fps, (width, height))

    print(f"Analyzing sequence of {total_frames} frames...")

    current_f = 0
    highlight_frames_saved = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Clean copy of the frame for output
        clean_frame = frame.copy()
        kp_norm, _ = detector.get_poses(frame)
        
        if kp_norm is not None:
            sequence = processor.prepare_sequence(kp_norm)
            
            if sequence is not None:
                input_tensor = torch.FloatTensor(sequence).to(device)
                with torch.no_grad():
                    output = classifier(input_tensor)
                    prediction = torch.argmax(output, dim=1).item()

                    # --- HIGHLIGHT LOGIC ---
                    # Save frame if the model predicts Dyno (2) or Top (3)
                    if prediction == 2: 
                        cv2.putText(clean_frame, "HIGHLIGHT: DYNO", (50, 80), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 4)
                        out.write(clean_frame)
                        highlight_frames_saved += 1
                    elif prediction == 3:
                        cv2.putText(clean_frame, "HIGHLIGHT: TOP/FALL", (50, 80), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
                        out.write(clean_frame)
                        highlight_frames_saved += 1

        current_f += 1
        if current_f % 100 == 0:
            print(f"Progress: {current_f}/{total_frames} frames analyzed...")

    cap.release()
    out.release()
    
    # Summary
    if fps > 0:
        seconds_saved = highlight_frames_saved / fps
        print(f"Debrief video generated: {video_out}")
        print(f"Highlight duration: {seconds_saved:.1f} seconds retained.")
    else:
        print("Error: Could not read frames from the source video.")

if __name__ == "__main__":
    video_input = os.path.join(PROJECT_ROOT, "data", "input", "janja_sequence_318_385.mp4")
    video_output = os.path.join(PROJECT_ROOT, "data", "output", "janja_highlights_sequence_318_385.mp4")
    
    create_highlights_from_clip(video_input, video_output)