import torch
from src.classifier import ClimbingLSTM
import os

os.makedirs('models', exist_ok=True)
model = ClimbingLSTM()
torch.save(model.state_dict(), "models/action_lstm.pth")
print("File models/action_lstm.pth generated with random initialization weights.")