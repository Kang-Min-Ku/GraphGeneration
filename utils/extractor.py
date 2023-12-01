import torch
import os

def extract_feature(model,
                    train_loader, test_loader,
                    is_save=True, save_path="save", model_name="resnet", dataset_name="cifar"):
    model.eval()
    device = next(model.parameters()).device
    embedding = []
    label = []

    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images, is_extract=True)
            embedding.append(outputs)
            label.append(labels)

        print("Complete extracting train features")

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images, is_extract=True)
            embedding.append(outputs)
            label.append(labels)

        print("Complete extracting test features")

    embedding = torch.cat(embedding, dim=0)
    label = torch.cat(label, dim=0)
    _, embedding_size = embedding.shape

    if is_save:
        torch.save([embedding, label], os.path.join(save_path, f"{dataset_name}_{model_name}_emb_{embedding_size}.pt"))

    return embedding, label