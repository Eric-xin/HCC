import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# from CNN_model import CNNModel
from CNN_RNN_model import CNNRNNModel
from dataloader import dataloader

# Initialize TensorBoard writer
writer = SummaryWriter()

# Function to train the model
def train_model(model, dataloader, criterion, optimizer, num_epochs=10, device='cpu'):
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(tqdm(dataloader)):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Log the loss
            if i % 10 == 9:  # Log every 10 batches
                writer.add_scalar('training_loss', running_loss / 10, epoch * len(dataloader) + i)
                running_loss = 0.0
                writer.add_images('inputs', inputs[:4], epoch * len(dataloader) + i)
                writer.add_images('ground_truth', labels[:4], epoch * len(dataloader) + i)
                writer.add_images('predictions', outputs[:4], epoch * len(dataloader) + i)
                
            # Log predictions and ground truth images
            # if i % 100 == 99:  # Log every 100 batches
            #     writer.add_images('inputs', inputs[:4], epoch * len(dataloader) + i)
            #     writer.add_images('ground_truth', labels[:4], epoch * len(dataloader) + i)
            #     writer.add_images('predictions', outputs[:4], epoch * len(dataloader) + i)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
        # # Save the model checkpoint
        # torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')

    # Save the final model
    torch.save(model.state_dict(), '_model.pth')
    
    print('Finished Training')
    # writer.flush() # Flush the TensorBoard writer to write the logs to disk
    writer.close()

# Example usage
if __name__ == "__main__":
    # Device configuration
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model, criterion, and optimizer
    # model = CNNModel()
    model = CNNRNNModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    train_model(model, dataloader, criterion, optimizer, num_epochs=10, device=device)