import torch
import torch.nn as nn
import timm

class ResNetCifar100(nn.Module):
    def __init__(self,
                 embedding_dim,
                 is_freeze):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.pretrained = timm.create_model("resnet50_cifar100", pretrained=True) #huggingface/timm

        num_classes, hidden_dim = self.pretrained.fc.weight.shape
        self.pretrained.fc = torch.nn.Linear(hidden_dim, self.embedding_dim)
        if is_freeze:
            for name, param in self.pretrained.named_parameters():
                if not name.startswith("fc"):
                    param.requires_grad = False

        self.fc = torch.nn.Linear(self.embedding_dim, num_classes)

    def forward(self, x, is_extract=False):
        x = self.pretrained(x)
        if not is_extract:
            x = self.fc(x)
        return x
        