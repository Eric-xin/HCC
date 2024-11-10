import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json

import sys
import os
# set the path to root directory
# sys.path.append('..')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from CNN_model import CNNModel
from src.CNN_model import CNNModel
# try:
#     from utils import dataloader
# except ImportError:
#     from ..utils import dataloader
from utils import dataloader

def train_model(model, dataloader, device, learning_rate, num_epochs, loss_save, loss_dir, model_save, model_dir):
    model.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # List to store loss values
    loss_values = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss/len(dataloader))
        
        epoch_loss = running_loss / len(dataloader)
        loss_values.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Save the model
    if model_save:
        torch.save(model.state_dict(), model_dir)

    # Save the loss values to a file
    if loss_save:
        with open(loss_dir, 'w') as f:
            json.dump(loss_values, f)

    print("Finished Training, Model and Loss Values Saved")

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel()
    train_model(model=model,
                dataloader=dataloader,
                device=device,
                learning_rate=0.0005,
                num_epochs=35,
                loss_save=True,
                loss_dir='cache/loss_values.json',
                model_save=True,
                model_dir='cache/model.pth')