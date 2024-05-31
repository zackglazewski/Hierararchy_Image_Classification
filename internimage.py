import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torchvision.models as models
from transformers import AutoModel, ViTForImageClassification
from timm.loss import LabelSmoothingCrossEntropy
from torch import nn, optim

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the data transformations
data_transforms = {
    'train': transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter()]), p=0.1),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.RandomErasing(p=0.2, value='random')
    ]),
    'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]),
}


print(device)

# Path to your dataset
data_dir = 'inat2021birds/bird_train'

# Create the full dataset
full_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms['train'])
num_classes = len(full_dataset.classes)


# Reserve a percentage for validation
val_percentage = 0.2
val_size = int(val_percentage * len(full_dataset))
train_size = len(full_dataset) - val_size


train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

full_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms['val'])
_, val_dataset = random_split(full_dataset, [train_size, val_size])

# # Apply different transforms to training and validation datasets
# train_dataset.dataset.transform = data_transforms['train']
# val_dataset.dataset.transform = data_transforms['val']

# Create data loaders with pin_memory enabled
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)

# Initialize the custom model
# model = AutoModel.from_pretrained("OpenGVLab/internimage_s_1k_224", trust_remote_code=True, num_classes=num_classes, ignore_mismatched_sizes=True)
# model = AutoModel.from_pretrained("OpenGVLab/internimage_xl_1k_384", trust_remote_code=True)
# print(model)
# model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
# model = ViTForImageClassification.from_pretrained("facebook/vit-mae-base")
import timm
model = timm.create_model("hf_hub:timm/efficientnet_b3.ra2_in1k", pretrained=True)
for param in model.parameters():
    param.requires_grad = False

n_inputs = model.classifier.in_features

model.classifier = nn.Sequential(
    nn.Linear(n_inputs,2048),
    nn.SiLU(),
    nn.Dropout(0.3),
    nn.Linear(2048, num_classes)
)
print(model)
model.to(device)



# Define the loss function and optimizer

criterion = nn.CrossEntropyLoss()
# criterion = LabelSmoothingCrossEntropy()
criterion.to(device)

# 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, weight_decay=0.05)
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.97)
highest_accuracy = 0

from tqdm import tqdm
def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, "Training"):
        images, labels = images.to(device), labels.to(device)  # Move data to GPU
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    accuracy = correct / total
    return epoch_loss, accuracy

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, "Validating"):
            images, labels = images.to(device), labels.to(device)  # Move data to GPU
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    accuracy = correct / total
    return epoch_loss, accuracy

torch.cuda.empty_cache()

# Training the model
num_epochs = 100
for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    if (val_acc > highest_accuracy):
        highest_accuracy = val_acc
        torch.save(model.state_dict(), "inatmini_internimage_best_2.pth")
        print("Found new highest with accuracuy: ", highest_accuracy)

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), 'intern_image_5_27.pth')