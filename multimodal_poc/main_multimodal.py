import cv2
import torch
import numpy as np
import os
import sys
from moviepy import VideoFileClip, AudioFileClip

# Absolute paths configuration to ensure script runs from anywhere
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

from src.detector import ClimbingDetector
from src.processor import PoseProcessor
from src.audio_engine import ClimbingAudioEngine
from src.fusion_model import ClimbingMultimodalNet

def run_multimodal_inference():
    """
    Main execution pipeline for the Multimodal POC.
    Runs Vision (YOLO) and Audio (Librosa) simultaneously, passes them to the Late Fusion LSTM,
    and outputs an annotated MP4 file with remuxed audio.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Launching Multimodal Inference on: {device}")

    # 1. File Paths Configuration
    video_in = os.path.join(PROJECT_ROOT, "data", "input", "janja_sequence_318_385.mp4")
    audio_in = os.path.join(PROJECT_ROOT, "data", "input", "janja_audio_track.wav")
    
    # Output Files
    temp_video_out = os.path.join(PROJECT_ROOT, "data", "output", "temp_silent_analysis.mp4")
    final_video_out = os.path.join(PROJECT_ROOT, "data", "output", "janja_multimodal_analysis.mp4")
    os.makedirs(os.path.dirname(final_video_out), exist_ok=True)

    # 2. Load all engines
    yolo_path = os.path.abspath(os.path.join(PROJECT_ROOT, 'models', 'yolo11m-pose.pt'))
    if not os.path.exists(yolo_path):
        yolo_path = os.path.join(PROJECT_ROOT, 'models', 'yolo11m-pose.pt')

    lstm_path = os.path.join(PROJECT_ROOT, 'models', 'multimodal_net.pth')

    print("Loading models (Vision, Audio, and Fusion)...")
    detector = ClimbingDetector(yolo_path)
    processor = PoseProcessor(window_size=30)
    audio_engine = ClimbingAudioEngine(audio_in, fps=30)
    
    classifier = ClimbingMultimodalNet().to(device)
    # weights_only=True avoids arbitrary code execution warnings when loading PyTorch models
    classifier.load_state_dict(torch.load(lstm_path, map_location=device, weights_only=True))
    classifier.eval()

    # 3. Video Processing Configuration
    cap = cv2.VideoCapture(video_in)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_out, fourcc, fps, (width, height))

    print(f"Synchronized analysis running ({total_frames} frames)...")
    current_f = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # A. Vision: Skeleton extraction using YOLO
        kp_norm, results = detector.get_poses(frame)
        
        # Draw base skeleton if detected, otherwise keep original frame
        annotated_frame = results[0].plot() if results and len(results[0].boxes) > 0 else frame.copy()

        if kp_norm is not None:
            seq_vision = processor.prepare_sequence(kp_norm)
            
            # Once we have accumulated 1 second of visual history...
            if seq_vision is not None:
                # B. Audio: Retrieve corresponding 1-second audio window (MFCCs)
                seq_audio = audio_engine.get_audio_window(current_f, window_size=30)
                
                # C. Prepare tensors for the fusion
                t_vision = torch.FloatTensor(seq_vision).to(device)
                t_audio = torch.FloatTensor(np.expand_dims(seq_audio, axis=0)).to(device) # Add batch dimension

                # D. Multimodal Prediction
                with torch.no_grad():
                    output = classifier(t_vision, t_audio)
                    prediction = torch.argmax(output, dim=1).item()
                    action_label = classifier.classes[prediction]

                # E. Display Results & UI Overlay
                color = (0, 255, 0) # Green default (Rest / Climb)
                if prediction == 2: color = (0, 165, 255) # Orange for Dyno (Explosive move)
                elif prediction == 3: color = (255, 0, 255) # Purple for Top/Fall

                cv2.putText(annotated_frame, f"MULTIMODAL AI: {action_label}", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
                
                # Visual indicator that Audio Processing is actively contributing
                cv2.putText(annotated_frame, "Audio Engine: ACTIVE (MFCC)", (50, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        out.write(annotated_frame)
        current_f += 1
        
        if current_f % 100 == 0:
            print(f"Progress: {current_f}/{total_frames} frames processed.")

    cap.release()
    out.release()
    print("Visual and Neural analysis complete.")

    # 4. Final Output Generation: Audio Remuxing
    print("Remuxing audio track onto the final annotated video...")
    try:
        video_clip = VideoFileClip(temp_video_out)
        audio_clip = AudioFileClip(audio_in)
        
        # Subclip audio to match exact video duration and prevent MoviePy sync errors
        final_clip = video_clip.with_audio(audio_clip.subclipped(0, video_clip.duration))
        final_clip.write_videofile(final_video_out, codec="libx264", audio_codec="aac", logger=None)
        
        # Cleanup temporary muted file to save disk space
        video_clip.close()
        audio_clip.close()
        os.remove(temp_video_out)
        
        print(f"Final video with audio available at:\n -> {final_video_out}")
    except Exception as e:
        print(f"Warning: Silent video generated, but audio muxing failed: {e}")

if __name__ == "__main__":
    run_multimodal_inference()