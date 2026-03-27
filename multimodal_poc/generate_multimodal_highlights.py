"""
Multimodal Highlight Generator Module
Analyzes a full climbing sequence using both Vision and Audio (Late Fusion).
Extracts only the most spectacular moments (e.g., Dynos, Tops, or Falls) 
based on the Multimodal AI's predictions and outputs a condensed video reel WITH synchronized audio.
"""

import cv2
import torch
import numpy as np
import os
import sys
import soundfile as sf
from moviepy import VideoFileClip, AudioFileClip

# Absolute paths configuration to ensure script runs from anywhere
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

from src.detector import ClimbingDetector
from src.processor import PoseProcessor
from src.audio_engine import ClimbingAudioEngine
from src.fusion_model import ClimbingMultimodalNet

def create_multimodal_highlights(video_in, audio_in, video_out):
    """
    Reads video and audio inputs, runs synchronized multimodal inference on each frame, 
    and writes to the output video ONLY if the model predicts a high-intensity action.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating Multimodal Highlights on target device: {device}")

    # 1. Initialization and Paths
    yolo_path = os.path.abspath(os.path.join(PROJECT_ROOT, 'models', 'yolo11m-pose.pt'))
    if not os.path.exists(yolo_path):
        yolo_path = os.path.join(PROJECT_ROOT, 'models', 'yolo11m-pose.pt')
        
    lstm_path = os.path.join(PROJECT_ROOT, 'models', 'multimodal_net.pth')
    
    print("Loading Multimodal Models (Vision + Audio)...")
    detector = ClimbingDetector(yolo_path)
    processor = PoseProcessor(window_size=30)
    audio_engine = ClimbingAudioEngine(audio_in, fps=30)
    
    classifier = ClimbingMultimodalNet().to(device)
    # weights_only=True to prevent arbitrary code execution warnings
    classifier.load_state_dict(torch.load(lstm_path, map_location=device, weights_only=True))
    classifier.eval()

    # 2. Video Source Configuration
    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        print(f"ERROR: Cannot open source video at {video_in}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 3. Output Configuration (Temporary files for muxing)
    os.makedirs(os.path.dirname(video_out), exist_ok=True)
    temp_video_out = video_out.replace(".mp4", "_temp.mp4")
    temp_audio_out = video_out.replace(".mp4", "_temp.wav")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(temp_video_out, fourcc, fps, (width, height))

    print(f"Analyzing full multimodal sequence of {total_frames} frames...")

    current_f = 0
    highlight_frames_saved = 0
    highlight_audio_chunks = [] # Stores audio slices

    # 4. Main Inference Loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Keep a clean copy to avoid drawing the raw YOLO skeleton on the highlight reel
        clean_frame = frame.copy()
        
        # A. Vision Extraction
        kp_norm, _ = detector.get_poses(frame)
        
        if kp_norm is not None:
            seq_vision = processor.prepare_sequence(kp_norm)
            
            # If we have gathered enough history (30 frames)
            if seq_vision is not None:
                
                # B. Audio Extraction (Synchronized)
                seq_audio = audio_engine.get_audio_window(current_f, window_size=30)
                
                # C. Tensor Preparation
                t_vision = torch.FloatTensor(seq_vision).to(device)
                t_audio = torch.FloatTensor(np.expand_dims(seq_audio, axis=0)).to(device)
                
                # D. Neural Prediction
                with torch.no_grad():
                    output = classifier(t_vision, t_audio)
                    prediction = torch.argmax(output, dim=1).item()

                    # --- HIGHLIGHT FILTERING LOGIC ---
                    if prediction in [2, 3]: # 2 = Dyno, 3 = Top/Fall
                        if prediction == 2:
                            cv2.putText(clean_frame, "HIGHLIGHT: DYNO", (50, 80), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 4)
                        elif prediction == 3:
                            cv2.putText(clean_frame, "HIGHLIGHT: TOP/FALL", (50, 80), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
                        
                        out.write(clean_frame)
                        highlight_frames_saved += 1
                        
                        # --- AUDIO STITCHING LOGIC ---
                        # Extract the exact fraction of audio corresponding to this frame
                        start_sample = int((current_f / fps) * audio_engine.sr)
                        end_sample = int(((current_f + 1) / fps) * audio_engine.sr)
                        audio_chunk = audio_engine.y[start_sample:end_sample]
                        
                        # Avoid appending empty arrays at the very edge of the video
                        if len(audio_chunk) > 0:
                            highlight_audio_chunks.append(audio_chunk)

        current_f += 1
        if current_f % 100 == 0:
            print(f"Progress: {current_f}/{total_frames} frames analyzed...")

    # 5. Cleanup
    cap.release()
    out.release()
    
    # 6. Audio Remuxing and Final Output
    if highlight_frames_saved > 0 and len(highlight_audio_chunks) > 0:
        print("Stitching highlight audio chunks together...")
        final_audio_raw = np.concatenate(highlight_audio_chunks)
        
        # Convert Mono to Stereo to avoid AAC encoding issues in MoviePy
        if len(final_audio_raw.shape) == 1:
            final_audio_raw = np.column_stack((final_audio_raw, final_audio_raw))
            
        # Ensure it writes as standard 16-bit PCM WAV
        sf.write(temp_audio_out, final_audio_raw, audio_engine.sr, subtype='PCM_16')
        
        print("Remuxing audio and video into final highlight reel...")
        try:
            v_clip = VideoFileClip(temp_video_out)
            a_clip = AudioFileClip(temp_audio_out)
            
            # Combine the stitched video with the stitched audio
            final_clip = v_clip.with_audio(a_clip)
            
            # FIX: Explicitly set audio_fps=44100. 16kHz AAC is often muted by media players!
            final_clip.write_videofile(
                video_out, 
                codec="libx264", 
                audio_codec="aac", 
                audio_fps=44100, 
                logger=None
            )
            
            # Clean up temporary files
            v_clip.close()
            a_clip.close()
            os.remove(temp_video_out)
            os.remove(temp_audio_out)
            
            seconds_saved = highlight_frames_saved / fps
            print(f"Debrief video generated successfully at: {video_out}")
            print(f"Highlight reel duration: {seconds_saved:.1f} seconds of pure action (with sound!).")
            
        except Exception as e:
            print(f"Muxing failed: {e}")
    else:
        print("No highlights were found. The output video is empty.")

if __name__ == "__main__":
    # Define IO paths relative to the project root
    video_input = os.path.join(PROJECT_ROOT, "data", "input", "janja_sequence_318_385.mp4")
    audio_input = os.path.join(PROJECT_ROOT, "data", "input", "janja_audio_track.wav")
    video_output = os.path.join(PROJECT_ROOT, "data", "output", "janja_multimodal_highlights.mp4")
    
    create_multimodal_highlights(video_input, audio_input, video_output)