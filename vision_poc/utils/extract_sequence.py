import cv2
import os

def extract_clip(input_path, output_path, start_sec, end_sec):
    print(f"Cutting video from {start_sec}s to {end_sec}s")
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    
    # Seek to the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Output file configuration
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    current_frame = start_frame
    total_frames = end_frame - start_frame
    
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret: break
            
        out.write(frame)
        current_frame += 1
        
        if (current_frame - start_frame) % 100 == 0:
            print(f"Progress: {current_frame - start_frame}/{total_frames} frames extracted.")
            
    cap.release()
    out.release()
    print(f"Original sequence saved to: {output_path}")

if __name__ == "__main__":
    # PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
    PROJECT_ROOT = os.getcwd()

    # Full source video
    video_in = os.path.join(PROJECT_ROOT, "data", "input", "janja_video.mp4")
    
    # New short video clip
    start_sec = 318
    end_sec = 385

    video_out = os.path.join(
        PROJECT_ROOT,
        "data",
        "input",
        f"janja_sequence_{start_sec}_{end_sec}.mp4"
    )
    
    extract_clip(video_in, video_out, start_sec, end_sec)