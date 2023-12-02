import torch
import torch.nn as nn
import timm
from transformers import AutoImageProcessor, AutoModelForImageClassification

class VitCifar100(nn.Module):
    """
    resize is necessary
    """
    def __init__(self,
                 embedding_dim,
                 is_freeze):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.pretrained = timm.create_model("timm/vit_base_patch16_224.orig_in21k_ft_in1k", pretrained=False)
        self.pretrained.head = nn.Linear(self.pretrained.head.in_features, 100)
        self.pretrained.load_state_dict(
            torch.hub.load_state_dict_from_url(
                "https://huggingface.co/edadaltocg/vit_base_patch16_224_in21k_ft_cifar100/resolve/main/pytorch_model.bin",
                map_location="cpu",
                file_name="vit_base_patch16_224_in21k_ft_cifar100.pth",
            )
        )
        
        num_classes, hidden_dim = self.pretrained.head.weight.shape
        self.pretrained.head = torch.nn.Linear(hidden_dim, self.embedding_dim)
        if is_freeze:
            for name, param in self.pretrained.named_parameters():
                if not name.startswith("head"):
                    param.requires_grad = False

        self.fc = torch.nn.Linear(self.embedding_dim, num_classes)

    def forward(self, x, is_extract=False):
        x = self.pretrained(x)
        if not is_extract:
            x = self.fc(x)
        return x    
    
class VitDog(nn.Module):
    def __init__(self,
                 embedding_dim,
                 is_freeze):
        super().__init__()

        self.embedding_dim = embedding_dim
        processor = AutoImageProcessor.from_pretrained("ep44/Stanford_dogs-google_vit_base_patch16_224")
        self.pretrained = AutoModelForImageClassification.from_pretrained("ep44/Stanford_dogs-google_vit_base_patch16_224")
        
        num_classes, hidden_dim = self.pretrained.classifier.weight.shape
        self.pretrained.classifier = torch.nn.Linear(hidden_dim, self.embedding_dim)
        if is_freeze:
            for name, param in self.pretrained.named_parameters():
                if not name.startswith("classifier"):
                    param.requires_grad = False

        self.fc = torch.nn.Linear(self.embedding_dim, num_classes)

    def forward(self, x, is_extract=False):
        x = self.pretrained(x)
        x = x.logits
        if not is_extract:
            x = self.fc(x)
        return x