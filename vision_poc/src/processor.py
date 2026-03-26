import numpy as np

class PoseProcessor:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.pose_history = []

    def prepare_sequence(self, keypoints):
        """
        Flattens the 17 (x,y) keypoints and manages the rolling temporal window.
        Returns a sequence of shape (1, 30, 34) when the window is full.
        """
        # Flatten coordinates: [x1, y1, x2, y2, ...] -> 34 features
        flattened_kp = keypoints.flatten()
        
        # Ensure we have exactly 34 values (17 points * 2 coordinates)
        if len(flattened_kp) != 34:
            return None

        self.pose_history.append(flattened_kp)
        
        # Maintain the rolling window size
        if len(self.pose_history) > self.window_size:
            self.pose_history.pop(0)
            return np.array([self.pose_history]) # Shape suitable for LSTM: (1, 30, 34)
        
        return None