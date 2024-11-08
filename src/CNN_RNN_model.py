import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNRNNModel(nn.Module):
    def __init__(self, rnn_hidden_size=128, rnn_num_layers=2):
        super(CNNRNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # RNN parameters
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        
        # Fully connected layer to reduce the dimension before feeding into RNN
        self.fc1 = nn.Linear(128 * 37 * 37, 512)
        
        # RNN layer
        self.rnn = nn.LSTM(input_size=512, hidden_size=rnn_hidden_size, num_layers=rnn_num_layers, batch_first=True)
        
        # Fully connected layer for the output
        self.fc2 = nn.Linear(rnn_hidden_size, 2)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the output from the convolutional layers
        x = x.view(-1, 128 * 37 * 37)
        
        # Pass through the fully connected layer
        x = F.relu(self.fc1(x))
        
        # Reshape for RNN input (batch_size, seq_length, input_size)
        x = x.unsqueeze(1)  # Adding sequence length dimension
        
        # Pass through the RNN
        rnn_out, _ = self.rnn(x)
        
        # Take the output of the last time step
        rnn_out = rnn_out[:, -1, :]
        
        # Pass through the final fully connected layer
        x = self.fc2(rnn_out)
        
        return x

model = CNNRNNModel()