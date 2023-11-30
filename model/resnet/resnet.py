import torch
from ..pretrained import load_pretrained

class ResNet(torch.nn.Module):
    def __init__(self, dataset, embedding_dim, is_freeze=True):
        super().__init__()

        assert dataset in ["cifar100", "dog"], "dataset must be either cifar100 or dog"

        self.dataset = dataset
        self.embedding_dim = embedding_dim
        self.pretrained = load_pretrained(dataset, "resnet")
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
        