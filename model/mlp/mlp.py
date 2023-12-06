import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)  # Add batch normalization layer
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)  # Add batch normalization layer
        self.relu2 = nn.ReLU()
        self.classifier = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)  # Apply batch normalization
        out = self.relu1(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.bn2(out)  # Apply batch normalization
        out = self.relu2(out)
        out = self.dropout(out)
        
        out = self.classifier(out)
        
        return out