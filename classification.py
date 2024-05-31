from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from tqdm import tqdm
torch.manual_seed(0)
import random
random.seed(0)
# training gets slightly corrupted by flipping horizontally, to get more training data
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Download and create the full training dataset
#full_dataset = datasets.INaturalist(root='data', version='2021_train_mini', download=False, transform=train_transforms)
full_dataset = datasets.ImageFolder(root="data_mini/2021_train_mini", transform=train_transforms)

# Reserve a percentage for validation
val_percentage = 0.2
val_size = int(val_percentage * len(full_dataset))
train_size = len(full_dataset) - val_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create data loaders with pin_memory enabled
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#all_labels = np.array([sample[1] for sample in full_dataset])


#num_classes = len(np.unique(all_labels))
num_classes = 2233

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)

for param in model.parameters():
    param.requires_grad = False

model.fc.requires_grad = True
print("FC layer requires_grad:", model.fc.weight.requires_grad)
print(model)
model.to(device)

criterion = nn.CrossEntropyLoss()
#criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss=0.0
    correct=0
    total=0

    for images, labels in tqdm(loader,  "Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs=model(images)
        print("Check: ", outputs.requires_grad)
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
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    accuracy = correct / total
    return epoch_loss, accuracy

num_epochs = 100

current_accuracy = 0
for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    if (val_acc > current_accuracy):
        print("Achieved new maximum validation accuracy")
        torch.save(model.state_dict(), "resnet50_freeze_5_25_best.pth")

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    torch.save(model.state_dict(), "resnet50_freeze_5_25.pth")