import torch
import torch.nn as nn

class ClimbingLSTM(nn.Module):
    def __init__(self, input_dim=34, hidden_dim=64, num_classes=4):
        super(ClimbingLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
        # English class labels for the UI
        self.classes = ["Rest", "Climb", "Dyno (Jump)", "Fall/Top"]

    def forward(self, x):
        # x shape: (Batch, 30 frames, 34 features)
        lstm_out, _ = self.lstm(x)
        
        # We only need the output from the very last time-step to make a prediction
        last_time_step = lstm_out[:, -1, :]
        return self.fc(last_time_step)