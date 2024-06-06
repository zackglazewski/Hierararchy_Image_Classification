import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split
from torchvision import models

import timm
from timm.loss import LabelSmoothingCrossEntropy
from timm.data import create_transform

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys
from tqdm import tqdm
import time
import copy

def formatText(class_label):
    return " ".join(class_label.split("_")[-2:])

def get_classes(data_dir):
    all_data = datasets.ImageFolder(data_dir)
    # print("Classes: ")
    # print(all_data.classes)
    # print()
    return all_data.classes





def get_data_loaders(data_dir, batch_size, train = False):
    if train:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter()]), p=0.1),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.RandomErasing(p=0.2, value='random')
        ])
        all_data = datasets.ImageFolder(data_dir, transform=transform)
        train_data_len = int(len(all_data)*0.78)
        valid_data_len = int((len(all_data) - train_data_len)/2)
        test_data_len = int(len(all_data) - train_data_len - valid_data_len)
        train_data, val_data, test_data = random_split(all_data, [train_data_len, valid_data_len, test_data_len])
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
        return train_loader, train_data_len

    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        all_data = datasets.ImageFolder(data_dir, transform=transform)
        train_data_len = int(len(all_data)*0.78)
        valid_data_len = int((len(all_data) - train_data_len)/2)
        test_data_len = int(len(all_data) - train_data_len - valid_data_len)
        train_data, val_data, test_data = random_split(all_data, [train_data_len, valid_data_len, test_data_len])
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)
        return (val_loader, test_loader, valid_data_len, test_data_len)
    
dataset_path = "inat2021birds/bird_train"
(train_loader, train_data_len) = get_data_loaders(dataset_path, 32, train=True)
(val_loader, test_loader, valid_data_len, test_data_len) = get_data_loaders(dataset_path, 32, train=False)
classes = get_classes(dataset_path)

dataloaders = {
    "train":train_loader,
    "val": val_loader
}
dataset_sizes = {
    "train":train_data_len,
    "val": valid_data_len
}



print("Before transform: ", classes[0])
test = formatText(classes[0])
print("After transform: " , test)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
torch.backends.cudnn.benchmark = True

# model = models.efficientnet_b3(pretrained=True)
# model = timm.create_model("tf_efficientnetv2_b0", pretrained=True)
# model = timm.create_model("hf_hub:timm/efficientnet_b3.ra2_in1k", pretrained=True)
# # model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)




model = timm.create_model("hf_hub:timm/efficientnet_b3.ra2_in1k", pretrained=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)



# freeze weights
for param in model.parameters():
    param.requires_grad = False

n_inputs = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Linear(n_inputs,2048),
    nn.SiLU(),
    nn.Dropout(0.3),
    nn.Linear(2048, len(classes))
)

model = model.to(device)
criterion = criterion.to(device)

training_history = {'accuracy':[],'loss':[]}
validation_history = {'accuracy':[],'loss':[]}


def train_model(model, criterion, optimizer, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)


                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            # if phase == 'train':
            #     # scheduler.step()
            #     pass

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                training_history['accuracy'].append(epoch_acc)
                training_history['loss'].append(epoch_loss)
            elif phase == 'val':
                validation_history['accuracy'].append(epoch_acc)
                validation_history['loss'].append(epoch_loss)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    

train_model(model, criterion, optimizer, num_epochs=100)

# torch.cuda.empty_cache()
# model.load_state_dict()
# from tqdm import tqdm
# test_loss = 0.0
# class_correct = list(0. for i in range(len(classes)))
# class_total = list(0. for i in range(len(classes)))

# model_ft.eval()

# for data, target in tqdm(test_loader):
#     if torch.cuda.is_available():
#         data, target = data.cuda(), target.cuda()
#     with torch.no_grad():
#         output = model_ft(data)
#         loss = criterion(output, target)
#     test_loss += loss.item()*data.size(0)
#     _, pred = torch.max(output, 1)
#     correct_tensor = pred.eq(target.data.view_as(pred))
#     correct = np.squeeze(correct_tensor.numpy()) if not torch.cuda.is_available() else np.squeeze(correct_tensor.cpu().numpy())
#     if len(target) == 128:
#       for i in range(128):
#           label = target.data[i]
#           class_correct[label] += correct[i].item()
#           class_total[label] += 1

# test_loss = test_loss/len(test_loader.dataset)
# print('Test Loss: {:.6f}\n'.format(test_loss))

# for i in range(len(classes)):
#     if class_total[i] > 0:
#         print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
#             formatText(classes[i]), 100 * class_correct[i] / class_total[i],
#             np.sum(class_correct[i]), np.sum(class_total[i])))
#     else:
#         print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

# print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
#     100. * np.sum(class_correct) / np.sum(class_total),
#     np.sum(class_correct), np.sum(class_total)))

# example = torch.rand(1, 3, 224, 224)
# traced_script_module = torch.jit.trace(model_ft.cpu(), example)
# traced_script_module.save("birds-without-lca.pth")

# print("exp1 accuracy")
# print(validation_history['accuracy'])
# print()
# print("exp1 loss")
# print(validation_history['loss'])
# print()

# exp2 = copy.deepcopy(validation_history)

# plot experiment 2
# epochs = list(range(1, 15))
# plt.figure()
# plt.plot(epochs, validation_history['accuracy'], marker='o', linestyle='-', color='b', label='Accuracy')
# plt.plot(epochs, validation_history['loss'], marker='x', linestyle='--', color='r', label='Loss')
# plt.title('Accuracy and Loss over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Value')
# plt.legend()

# # Save the plot to a file
# plt.savefig('experiment1_with_lca_addition.png')

# # plot both together
# epochs = list(range(1, 15))
# plt.figure()
# plt.plot(epochs, exp1['accuracy'], marker='o', linestyle='-', color='b', label='with lca')
# plt.plot(epochs, exp2['accuracy'], marker='x', linestyle='--', color='r', label='without lca')
# plt.title('With and Without LCA over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Value')
# plt.legend()