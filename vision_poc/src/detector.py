from ultralytics import YOLO

class ClimbingDetector:
    def __init__(self, model_path='models/yolo11m-pose.pt'):
        # Automatically downloads the YOLO model if it doesn't exist locally
        self.model = YOLO(model_path)

    def get_poses(self, frame):
        # Run inference with tracking enabled. Verbose=False to keep console clean
        results = self.model.track(frame, persist=True, verbose=False)
        
        # Check if results exist AND if any keypoints are detected in the frame
        if results and len(results[0].keypoints.data) > 0:
            try:
                # Return normalized (0.0 to 1.0) coordinates for the first detected person
                return results[0].keypoints.xyn[0].cpu().numpy(), results
            except:
                return None, None
        return None, None