# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

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

torchvision.__version__, torch.__version__ # ('0.11.2+cu102', '1.10.1+cu102')

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys
from tqdm import tqdm
import time
import copy
from tree import *

experiment_path = "./crossEntropyRelu/"

def count_equal(l1,l2):
    # print("l1: ", l1)
    # print("l2: ", l2)
    assert len(l1)==len(l2)
    total = 0
    for i1, i2 in zip(l1, l2):
        if (i1 == i2):

            total += 1
    # print("total: ", total)
    return total

def get_classes(data_dir):
    all_data = datasets.ImageFolder(data_dir)
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
    
dataset_path = "../inat2021birds/bird_train_100"
(train_loader, train_data_len) = get_data_loaders(dataset_path, 32, train=True)
(val_loader, test_loader, valid_data_len, test_data_len) = get_data_loaders(dataset_path, 32, train=False)
classes = get_classes(dataset_path)
print("num classes: {}".format(len(classes)))
taxonomy = Tree(classes)
print("num edges: {}".format(taxonomy.get_num_edges()))
dataloaders = {
    "train":train_loader,
    "val": val_loader
}
dataset_sizes = {
    "train":train_data_len,
    "val": valid_data_len
}

print(len(train_loader))
print(len(val_loader))
print(len(test_loader))

print(train_data_len, test_data_len, valid_data_len)

dataiter = iter(train_loader)
images, labels = next(dataiter)
images = images.numpy() # convert images to numpy for display


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

torch.backends.cudnn.benchmark = True

# model = models.efficientnet_b3(pretrained=True)
model = timm.create_model("tf_efficientnetv2_b0", pretrained=True)
# model = timm.create_model("hf_hub:timm/tf_efficientnet_lite0.in1k", pretrained=True)
# model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)

for param in model.parameters():
    param.requires_grad = False

n_inputs = model.classifier.in_features
print("n_input: ", n_inputs)
exit()
# TODO: 
model.classifier = nn.Sequential(
    nn.Linear(n_inputs,2048),
    nn.SiLU(),
    nn.Dropout(0.3),
    nn.Linear(2048, len(classes))
)
model = model.to(device)
print(model.classifier)

# TODO:
# criterion = nn.BCEWithLogitsLoss()
# criterion = LabelSmoothingCrossEntropy()
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

training_history = {'accuracy':[],'loss':[]}
validation_history = {'accuracy':[],'loss':[]}

# exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.97)

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
                
                # TODO:
                # target_strings, target_edges = taxonomy.get_target_path_batched(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    # print("outputs shape: ", outputs.shape)
                    # print("label shape: ", target_edges.shape)
                    # TODO:
                    # target_edges = target_edges.to(device)

                    # TODO:
                    # loss = criterion(outputs, target_edges)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # TODO:
                # final_predictions = taxonomy.interpret_prediction_greedy_batched(outputs)[0]
                # statistics
                running_loss += loss.item() * inputs.size(0)

                # TODO:
                running_corrects += torch.sum(preds == labels.data)
                # running_corrects += count_equal(final_predictions, target_strings)
            # if phase == 'train':
                # scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = float(running_corrects) / dataset_sizes[phase]
            
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

                torch.save(model.state_dict(), "{}best.pth".format(experiment_path))

        print()

        torch.save(model.state_dict(), "{}checkpoints/checkpoint{}.pth".format(experiment_path, epoch))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

model_ft = train_model(model, criterion, optimizer,
                       num_epochs=100)

torch.save(model.state_dict(), "{}last.pth".format(experiment_path))

torch.cuda.empty_cache()

from tqdm import tqdm
test_loss = 0.0
class_correct = list(0. for i in range(len(classes)))
class_total = list(0. for i in range(len(classes)))

model_ft.eval()

total_correct = 0
for data, target in tqdm(test_loader):
    if torch.cuda.is_available(): 
        data, target = data.cuda(), target.cuda()
    with torch.no_grad():
        output = model_ft(data)

        target_strings, target_edges = taxonomy.get_target_path_batched(target)

    final_predictions = taxonomy.interpret_prediction_greedy_batched(output)

    total_correct += count_equal(final_predictions, target_strings)

epoch_acc = float(total_correct) / test_data_len
print('Testing Acc: {:.4f}'.format(epoch_acc))