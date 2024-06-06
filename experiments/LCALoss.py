from torch import nn, sigmoid
import torch
import queue
import copy
from collections import deque
import numpy as np
torch.autograd.set_detect_anomaly(True)

from tree import *

class LCAEdgeLoss(nn.Module):
    def __init__(self, classes, clamp_loss=False):
        super(LCAEdgeLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.tree = Tree(classes)
        self.apply_max = clamp_loss

    def forward(self, outputs, targets):
        loss = self.bce_loss(outputs, targets)
        for batch_index in range(loss.shape[0]):
            for loss_index in range(loss.shape[1]):
                if (targets[batch_index][loss_index] == 0):
                    # diverged
                    par_index = self.tree.get_edge_parent(loss_index)
                    if (par_index != None):
                        loss[batch_index][loss_index] = loss[batch_index][loss_index] + loss[batch_index][par_index]
                else:
                    # on the right path, apply max operation
                    par_index = self.tree.get_edge_parent(loss_index)
                    if (par_index != None):
                        if (loss[batch_index][loss_index] < loss[batch_index][par_index]):
                            loss[batch_index][loss_index] = loss[batch_index][par_index]

        return loss.mean()

class LCAHeavyChildLoss(nn.Module):
    def __init__(self, classes, clamp_loss=False, greedy=True):
        super(LCAHeavyChildLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.tree = Tree(classes)
        self.apply_max = clamp_loss
        self.greedy = greedy

    def forward(self, outputs, targets):

        # print("before output: ", outputs)
        if (self.greedy):
            output = self.tree.interpret_prediction_greedy_batched(outputs)[1]
        else:
            output = self.tree.max_probability_path_batched(outputs)[1]
        # output = self.tree.max_probability_path_batched(outputs)[1]
        # print("output: ", output)
        # output: [ probability distribution ]
        # outputs: [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        # targets: [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0]

        # loss:    [l, l, l, l, l, l, l, l, l, l, l, l, l, l]
        # updated: [l+p, l, l+p, l, l, l, l+p, l, l, l, l, l, l, l]


        loss = self.bce_loss(outputs, targets)
        for batch_index in range(loss.shape[0]):
            for loss_index in range(loss.shape[1]):
                if ((output[batch_index][loss_index] == 1) and (targets[batch_index][loss_index] == 0)):
                    # diverged
                    par_index = self.tree.get_edge_parent(loss_index)
                    if (par_index != None):
                        loss[batch_index][loss_index] = loss[batch_index][loss_index] + loss[batch_index][par_index]
                elif (self.apply_max):
                    # on the right path, apply max operation
                    par_index = self.tree.get_edge_parent(loss_index)
                    if (par_index != None):
                        if (loss[batch_index][loss_index] < loss[batch_index][par_index]):
                            loss[batch_index][loss_index] = loss[batch_index][par_index]

        return loss.mean()
    
class LCAHeavyParentLoss(nn.Module):
    def __init__(self, classes, greedy=True):
        super(LCAHeavyParentLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.tree = Tree(classes)
        self.greedy = greedy

    def forward(self, outputs, targets):

        # print("before output: ", outputs)
        if (self.greedy):
            output = self.tree.interpret_prediction_greedy_batched(outputs)[1]
        else:
            output = self.tree.max_probability_path_batched(outputs)[1]
        # print("output: ", output)
        # output: [ probability distribution ]
        # outputs: [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        # targets: [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0]

        # loss:    [l, l, l, l, l, l, l, l, l, l, l, l, l, l]
        # updated: [l+p, l, l+p, l, l, l, l+p, l, l, l, l, l, l, l]


        loss = self.bce_loss(outputs, targets)
        for batch_index in range(loss.shape[0]):
            for loss_index in range(loss.shape[1]-1, -1, -1):
                if ((output[batch_index][loss_index] == 1) and (targets[batch_index][loss_index] == 0)):
                    # diverged
                    par_index = self.tree.get_edge_parent(loss_index)
                    if (par_index != None):
                        loss[batch_index][par_index] = loss[batch_index][loss_index] + loss[batch_index][par_index]
                

        return loss.mean()