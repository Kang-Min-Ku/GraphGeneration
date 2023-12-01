import timm
import detectors
import torch
import torch.nn as nn
from .resnet.resnet import ResNetCifar100
from .vit.vit import VitCifar100

def load(dataset,
        model,
        embedding_dim,
        device=torch.device("cuda"),
        is_precheck=False,
        test_loader=None,
        is_freeze=True):
    """
    return none if no pretrained model is available
    """
    assert dataset in ["cifar100", "dog"], "dataset must be either cifar100 or dog"

    pretrained = None
    if dataset == "cifar100":
        if model == "resnet":
            base = timm.create_model("resnet50_cifar100", pretrained=True)
            pretrained = ResNetCifar100(embedding_dim=embedding_dim,
                                        is_freeze=is_freeze)
        elif model == "vit":
            base = timm.create_model("timm/vit_base_patch16_224.orig_in21k_ft_in1k", pretrained=False)
            base.head = nn.Linear(base.head.in_features, 100)
            base.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    "https://huggingface.co/edadaltocg/vit_base_patch16_224_in21k_ft_cifar100/resolve/main/pytorch_model.bin",
                    map_location="cpu",
                    file_name="vit_base_patch16_224_in21k_ft_cifar100.pth",
                )
            )
            pretrained = VitCifar100(embedding_dim=embedding_dim,
                                     is_freeze=is_freeze)
    elif dataset == "dog":
        pass

    if is_precheck and test_loader is not None:
        base = base.to(device)
        base.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = base(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
        print("Accuracy of the network on the 10000 test images: %d %%" % (100 * correct / total))

    return pretrained