import torch
import torch.nn as nn

class ClimbingMultimodalNet(nn.Module):
    def __init__(self, vision_dim=34, audio_dim=13, hidden_v=64, hidden_a=32, num_classes=4):
        super(ClimbingMultimodalNet, self).__init__()
        
        # VISION BRANCH (Identical to POC 1)
        self.lstm_vision = nn.LSTM(input_size=vision_dim, 
                                   hidden_size=hidden_v, 
                                   num_layers=2, 
                                   batch_first=True)
        
        # AUDIO BRANCH (New)
        self.lstm_audio = nn.LSTM(input_size=audio_dim, 
                                  hidden_size=hidden_a, 
                                  num_layers=1,  # A lighter network is sufficient for audio
                                  batch_first=True) 
        
        # FUSION LAYER (Concatenation)
        # Input size is the sum of both branch outputs (64 + 32 = 96)
        fusion_dim = hidden_v + hidden_a
        
        self.fc_fusion = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2), # Prevents overfitting
            nn.Linear(32, num_classes)
        )
        
        # English class labels
        self.classes = ["Rest", "Climb", "Dyno", "Top/Fall"]

    def forward(self, x_vision, x_audio):
        """
        x_vision shape: (batch_size, 30_frames, 34_features)
        x_audio shape:  (batch_size, 30_frames, 13_features)
        """
        
        # 1. Vision Processing
        out_v, _ = self.lstm_vision(x_vision)
        last_step_v = out_v[:, -1, :] # Extract the last time-step (Shape: Batch, 64)
        
        # 2. Audio Processing
        out_a, _ = self.lstm_audio(x_audio)
        last_step_a = out_a[:, -1, :] # Extract the last time-step (Shape: Batch, 32) 
        
        # 3. Late Fusion via concatenation along the feature dimension (dim=1)
        # Shape: (Batch, 64 + 32) -> (Batch, 96)
        fused_features = torch.cat((last_step_v, last_step_a), dim=1) 
        
        # 4. Final Classification
        logits = self.fc_fusion(fused_features)
        
        return logits

# Quick dimensional consistency test
if __name__ == "__main__":
    model = ClimbingMultimodalNet()
    dummy_vision = torch.randn(1, 30, 34)
    dummy_audio = torch.randn(1, 30, 13)
    output = model(dummy_vision, dummy_audio)
    print("Multimodal Model ready!")
    print(f"Output shape: {output.shape} (Expected: 1 batch, 4 classes)")