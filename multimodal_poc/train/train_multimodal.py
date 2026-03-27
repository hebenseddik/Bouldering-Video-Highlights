import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import sys

# Absolute paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from src.fusion_model import ClimbingMultimodalNet

def train_multimodal():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Multimodal Training on: {device}")

    # 1. Load synchronized data
    dataset_dir = os.path.join(PROJECT_ROOT, 'data', 'dataset')
    
    try:
        X_vision_raw = np.load(os.path.join(dataset_dir, 'X_vision.npy'))
        X_audio_raw = np.load(os.path.join(dataset_dir, 'X_audio.npy'))
        y_raw = np.load(os.path.join(dataset_dir, 'y.npy'))
        print(f"Data loaded: Vision {X_vision_raw.shape} | Audio {X_audio_raw.shape}") 
    except FileNotFoundError:
        print(f"ERROR: Multimodal .npy files not found.")
        return

    # 2. Convert to PyTorch Tensors
    X_vision_tensor = torch.FloatTensor(X_vision_raw).to(device)
    X_audio_tensor = torch.FloatTensor(X_audio_raw).to(device)
    y_tensor = torch.LongTensor(y_raw).to(device)

    # Note: TensorDataset accepts multiple inputs
    dataset = TensorDataset(X_vision_tensor, X_audio_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 3. Initialize Fusion Network
    model = ClimbingMultimodalNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Lower LR for complex network

    epochs = 80

    print("Starting Multimodal Training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0

        # The loop extracts BOTH input tensors
        for batch_vision, batch_audio, batch_y in loader: 
            optimizer.zero_grad()
            
            # Model takes Vision AND Audio simultaneously
            predictions = model(batch_vision, batch_audio)
            
            loss = criterion(predictions, batch_y) 
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted_classes = torch.max(predictions, 1)
            correct += (predicted_classes == batch_y).sum().item()

        accuracy = 100 * correct / len(dataset)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss:.4f} | Accuracy: {accuracy:.2f}%")

    # 4. Save Multimodal Weights
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'multimodal_net.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Multimodal Model (Vision+Audio) trained and saved: {model_path}")

if __name__ == "__main__":
    train_multimodal()