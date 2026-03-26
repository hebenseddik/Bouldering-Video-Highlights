import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import sys

# 1. Absolute root path setup
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from src.classifier import ClimbingLSTM

def create_sequences(X, y, time_steps=30):
    """Transforms individual poses into temporal windows."""
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def train_overfit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # 2. Data loading
    dataset_dir = os.path.join(PROJECT_ROOT, 'data', 'dataset')
    
    try:
        X_raw = np.load(os.path.join(dataset_dir, 'X_raw.npy'))
        y_raw = np.load(os.path.join(dataset_dir, 'y_raw.npy'))
        print(f"Raw data loaded: {X_raw.shape}")
    except FileNotFoundError:
        print(f"ERROR: .npy files not found in {dataset_dir}")
        return

    # Security check
    if len(X_raw) == 0:
        print("ERROR: X_raw.npy is empty. Please run build_dataset.py first.")
        return

    # 3. Create temporal sequences (30 frames)
    X_seq, y_seq = create_sequences(X_raw, y_raw, time_steps=30)
    print(f"Sequences created: {X_seq.shape}")

    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X_seq).to(device)
    y_tensor = torch.LongTensor(y_seq).to(device)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 4. Model Initialization
    model = ClimbingLSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    epochs = 80 # Sufficient for overfitting on this specific sequence

    print("Starting training process...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0

        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted_classes = torch.max(predictions, 1)
            correct += (predicted_classes == batch_y).sum().item()

        accuracy = 100 * correct / len(dataset)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss:.4f} | Accuracy: {accuracy:.2f}%")

    # 5. Save the trained weights
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'action_lstm.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model trained and saved to {model_path}")

if __name__ == "__main__":
    train_overfit()