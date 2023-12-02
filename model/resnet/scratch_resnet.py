import torch
import torch.nn as nn
from torchvision.models import resnet50

class ResNet(nn.Module):
    def __init__(self,
                 embedding_dim,
                 num_classes):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.pretrained = resnet50(pretrained=True)
        _, hidden_dim = self.pretrained.fc.weight.shape
        self.pretrained.fc = torch.nn.Linear(hidden_dim, self.embedding_dim)

        self.fc = torch.nn.Linear(self.embedding_dim, num_classes)

    def forward(self, x, is_extract=False):
        x = self.pretrained(x)
        if not is_extract:
            x = self.fc(x)
        return x