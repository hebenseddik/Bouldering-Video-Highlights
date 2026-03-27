import os
import sys
from moviepy import VideoFileClip

def extract_audio_segment(original_mp4, wav_path, start_sec, end_sec):
    print(f"Opening ORIGINAL video: {original_mp4}")
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    
    try:
        video = VideoFileClip(original_mp4)
        
        if video.audio is None:
            print("Source video has no audio track! Check the file.")
            return

        # Temporal clipping
        print(f"Cutting audio from {start_sec}s to {end_sec}s...")
        audio_segment = video.audio.subclipped(start_sec, end_sec)
        
        # Save at 16kHz (Standard sample rate for AI)
        audio_segment.write_audiofile(wav_path, fps=16000, logger=None)
        print(f"Audio file successfully generated: {wav_path}")
        
    except Exception as e:
        print(f"Extraction error: {e}")
        
    finally:
        if 'video' in locals():
            video.close()

if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    video_originale = os.path.abspath(os.path.join(PROJECT_ROOT, "data", "input", "janja_video.mp4"))
    audio_out = os.path.join(PROJECT_ROOT, "data", "input", "janja_audio_track.wav")
    
    extract_audio_segment(video_originale, audio_out, 318, 386)