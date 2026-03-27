import librosa
import numpy as np
import os

class ClimbingAudioEngine:
    def __init__(self, audio_path, fps=30, sr=16000, n_mfcc=13):
        """
        Initializes the audio engine and precomputes features for the entire sequence.
        - sr=16000: Standard sampling rate for voice/audio AI.
        - n_mfcc=13: Number of frequency bands analyzed (lightweight and efficient).
        """ 
        self.fps = fps
        self.sr = sr
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        print("Loading audio and extracting MFCC frequencies...")
        self.y, _ = librosa.load(audio_path, sr=self.sr)
        
        # MAGIC SYNCHRONIZATION: 
        # Calculate 'hop_length' so Librosa outputs exactly 30 frames per second
        self.hop_length = int(self.sr / self.fps) # 16000 / 30 ≈ 533 samples/frame
        
        # MFCC Extraction
        mfcc_raw = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=n_mfcc, hop_length=self.hop_length)
        
        # Transpose to get a matrix shape: (Number_of_Frames, 13_Features)
        self.mfcc_features = mfcc_raw.T 
        print(f"Audio Engine ready! ({self.mfcc_features.shape[0]} audio frames generated)")

    def get_audio_window(self, current_frame, window_size=30):
        """
        Retrieves the audio window for the last 30 frames (1 second of sound).
        Works exactly like PoseProcessor for vision.
        """
        start_frame = current_frame - window_size
        
        # Handle the beginning of the video (padding with silence)
        if start_frame < 0:
            pad = np.zeros((-start_frame, self.mfcc_features.shape[1]))
            feat = self.mfcc_features[0:current_frame] if current_frame > 0 else np.empty((0, self.mfcc_features.shape[1]))
            return np.vstack((pad, feat))
            
        # Handle the end of the video (edge case alignment)
        elif current_frame > len(self.mfcc_features):
            return self.mfcc_features[-window_size:]
            
        else:
            # Normal case: return exactly 1 second of audio data
            return self.mfcc_features[start_frame:current_frame]