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
import random

experiment_path = "./heavyChild/"

torch.manual_seed(0)
random.seed(0)

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

torch.cuda.empty_cache()

from tqdm import tqdm
test_loss = 0.0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
# model = models.efficientnet_b3(pretrained=True)
model = timm.create_model("tf_efficientnetv2_b0", pretrained=True)
# model = timm.create_model("hf_hub:timm/tf_efficientnet_lite0.in1k", pretrained=True)
# model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)

for param in model.parameters():
    param.requires_grad = False

n_inputs = model.classifier.in_features

# TODO: 
model.classifier = nn.Sequential(
    nn.Linear(n_inputs,2048),
    nn.SiLU(),
    nn.Dropout(0.3),
    nn.Linear(2048, taxonomy.get_num_edges())
)
model.load_state_dict(torch.load("{}checkpoints/checkpoint99.pth".format(experiment_path)))
model.eval()
model = model.to(device)

total_correct = 0
for data, target in tqdm(test_loader):
    if torch.cuda.is_available(): 
        data, target = data.cuda(), target.cuda()
    with torch.no_grad():
        output = model(data)
        _, preds = torch.max(output, 1)

        # TODO:
        target_strings, target_edges = taxonomy.get_target_path_batched(target)

    final_predictions = taxonomy.interpret_prediction_greedy_batched(output)[0]
    # total_correct += torch.sum(preds == target.data)
    total_correct += count_equal(final_predictions, target_strings)

epoch_acc = float(total_correct) / test_data_len
print('Testing Acc: {:.4f}'.format(epoch_acc))