import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import re
import numpy as np
import timm
import detectors
import os
from data.loader import Loader
from utils.parser import YamlParser
from utils.extractor import extract_feature
from utils.logger import Logger
from transformers import EfficientNetForImageClassification, EfficientNetImageProcessor
from model.pretrained import load
from model.resnet.scratch_resnet import ResNet

logger = Logger()
logger.set_basic_config(filename="resnet50.log", level="INFO")
logger.set_logger(loggername="resnet50")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = YamlParser("hyperparam/base.yaml").args
loader = Loader("dataset", args, is_resize=True)
train_loader, test_loader = loader.load("dog")
num_class = 120
log_every = 300
test_every = 10

# Initialize the model
model = ResNet(args.embedding_size, num_class)
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

print("Setup complete. Training begins.")
logger.info("Setup complete. Training begins.")

# Training loop
total_step = len(train_loader)
for epoch in range(args.epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print training progress
        if (i+1) % log_every == 0:
            logger.info(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}")

    if (i+1) % test_every == 0:
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0

            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)

                # Get the predicted labels
                _, predicted = torch.max(outputs.data, 1)

                # Update the total and correct predictions
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            logger.info(f"Test Accuracy of the model on the {total} test images: {100 * correct / total}%")
        model.train()

print("Training complete. Testing begins.")
logger.info("Training complete. Testing begins.")

model.eval()
with torch.no_grad():
    correct = 0
    total = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        # Get the predicted labels
        _, predicted = torch.max(outputs.data, 1)

        # Update the total and correct predictions
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    logger.info(f"Test Accuracy of the model on the {total} test images: {100 * correct / total}%")

print("Testing complete. Saving model checkpoint.")
logger.info("Testing complete. Saving model checkpoint.")
# Save the model checkpoint
torch.save(model.state_dict(), 'save/resnet.pt')
print("Model checkpoint saved. Extracting features.")
logger.info("Model checkpoint saved. Extracting features.")
extract_feature(model, train_loader, test_loader, model_name=args.model, dataset_name=args.dataset)





