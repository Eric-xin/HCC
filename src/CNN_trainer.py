import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from CNN_model import CNNModel
from dataloader import dataloader

model = CNNModel()
dataloader = dataloader

# Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('mps' if torch.mps.is_available() else 'cpu')
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Hyperparameters
learning_rate = 0.001
num_epochs = 10

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)

        # # debug the model
        # print(f"outputs: {outputs.size()}, labels: {labels.size()}")
        # print(f"outputs: {outputs}, labels: {labels}")
        # # halt
        # break

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        progress_bar.set_postfix(loss=running_loss/len(dataloader))
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

# Save the model
torch.save(model.state_dict(), 'model.pth')
print("Finished Training and Model Saved")