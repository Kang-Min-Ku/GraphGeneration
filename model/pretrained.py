import timm
import detectors
import torch

def load_pretrained(dataset,
                    model,
                    check_model_performance=False,
                    test_loader=None,
                    device=torch.device("cuda:0")):
    """
    return none if no pretrained model is available
    """
    pretrained = None
    if dataset == "cifar100":
        if model == "resnet":
            pretrained = timm.create_model("resnet50_cifar100", pretrained=True)
    elif dataset == "dog":
        pass

    if check_model_performance and test_loader is not None:
        pretrained = pretrained.to(device)
        pretrained.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = pretrained(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
        print("Accuracy of the network on the 10000 test images: %d %%" % (100 * correct / total))

    pretrained = pretrained.to(torch.device("cpu"))

    return pretrained