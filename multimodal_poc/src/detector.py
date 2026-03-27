from ultralytics import YOLO

class ClimbingDetector:
    def __init__(self, model_path='yolo11m-pose.pt'):
        # Automatically downloads the model if it does not exist
        self.model = YOLO(model_path)

    def get_poses(self, frame):
        # Confidence threshold set to 0.6 to filter out false positives 
        results = self.model(frame, conf=0.6, verbose=False) 

        # Check if results exist AND if keypoints are detected
        if results and len(results[0].keypoints.data) > 0:
            try:
                # Return normalized skeleton coordinates (0.0 to 1.0)
                return results[0].keypoints.xyn[0].cpu().numpy(), results
            except:
                return None, None
        return None, None 