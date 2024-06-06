from torch import nn, sigmoid
import torch
import queue
import copy
from collections import deque
import numpy as np
torch.autograd.set_detect_anomaly(True)

def lca_distance(s1, s2):
    a_tokens = s1.split("_")
    b_tokens = s2.split("_")

    assert len(a_tokens) == len(b_tokens) , "lca_distance error: classes differ in token sizes"
    distance_to_root = len(a_tokens)

    
    
    distance_until_diverge = 0
    for a,b in zip(a_tokens, b_tokens):
        if (a==b):
            distance_until_diverge += 1
        else:
            break
    return distance_to_root - distance_until_diverge

def formatText(class_label):
    # return "_".join(class_label.split("_")[4:])
    # return "_".join(class_label.split("_")[6:])
    return "_".join(class_label.split("_")[1:])


class ExpLCACrossEntropy(nn.Module):
    def __init__(self, classes):
        super(ExpLCACrossEntropy, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.tree = classes

    def forward(self, outputs, targets):
        ce_loss = self.cross_entropy(outputs, targets)
        lca_loss = 0
        
        # Assuming targets and outputs are class indices
        for i in range(len(targets)):
            predicted_class = torch.argmax(outputs[i]).item()
            actual_class = targets[i].item()

            distance = lca_distance(formatText(self.tree[predicted_class]), formatText(self.tree[actual_class]))
            lca_loss += distance
        
        lca_loss /= len(targets)
        
        # Combine the losses
        total_loss = ce_loss + (2**(lca_loss))
        return total_loss
    
class LCA2CrossEntropy(nn.Module):
    def __init__(self, classes):
        super(LCA2CrossEntropy, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.tree = classes

    def forward(self, outputs, targets):
        ce_loss = self.cross_entropy(outputs, targets)
        lca_loss = 0
        
        # Assuming targets and outputs are class indices
        for i in range(len(targets)):
            predicted_class = torch.argmax(outputs[i]).item()
            actual_class = targets[i].item()

            distance = lca_distance(formatText(self.tree[predicted_class]), formatText(self.tree[actual_class]))
            lca_loss += (distance**2)
        
        lca_loss /= len(targets)
        
        # Combine the losses
        total_loss = ce_loss + lca_loss
        return total_loss
    
class LCA1CrossEntropy(nn.Module):
    def __init__(self, classes):
        super(LCA1CrossEntropy, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.tree = classes

    def forward(self, outputs, targets):
        ce_loss = self.cross_entropy(outputs, targets)
        lca_loss = 0

        # Assuming targets and outputs are class indices
        for i in range(len(targets)):
            predicted_class = torch.argmax(outputs[i]).item()
            actual_class = targets[i].item()

            distance = lca_distance(formatText(self.tree[predicted_class]), formatText(self.tree[actual_class]))
            lca_loss += distance
        
        lca_loss /= len(targets)
        
        # Combine the losses
        total_loss = ce_loss + lca_loss
        return total_loss

class LCA2Loss(nn.Module):
    def __init__(self, classes):
        super(LCA2Loss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.tree = classes

    def forward(self, outputs, targets):
        lca_loss = 0
        
        # Assuming targets and outputs are class indices
        for i in range(len(targets)):
            predicted_class = torch.argmax(outputs[i]).item()
            actual_class = targets[i].item()

            distance = lca_distance(formatText(self.tree[predicted_class]), formatText(self.tree[actual_class]))
            lca_loss += (distance**2)
        
        lca_loss = torch.tensor(lca_loss / len(targets), requires_grad=True)
        
        return lca_loss
    
class LCA1Loss(nn.Module):
    def __init__(self, classes):
        super(LCA1Loss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.tree = classes

    def forward(self, outputs, targets):
        lca_loss = 0
        
        # Assuming targets and outputs are class indices
        for i in range(len(targets)):
            predicted_class = torch.argmax(outputs[i]).item()
            actual_class = targets[i].item()

            distance = lca_distance(formatText(self.tree[predicted_class]), formatText(self.tree[actual_class]))


            lca_loss += distance
        
        lca_loss = torch.mean([lca_distance(formatText(prediction), formatText(truth)) for prediction, truth in zip()])
        
        lca_loss = torch.tensor(lca_loss / len(targets), requires_grad=True)
        
        return lca_loss
    
class LCALoss(nn.Module):
    def __init__(self, classes):
        super(LCALoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.tree = Tree(classes)

    def forward(self, outputs, targets):
        return self.bce_loss(outputs, targets)
    
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
    
class LCAPathLoss(nn.Module):
    def __init__(self, classes, clamp_loss=False):
        super(LCAPathLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.tree = Tree(classes)
        self.apply_max = clamp_loss

    def forward(self, outputs, targets):

        # print("before output: ", outputs)
        output = self.tree.interpret_batched_prediction_greedy(outputs)[1]
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
    def __init__(self, classes):
        super(LCAHeavyParentLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.tree = Tree(classes)

    def forward(self, outputs, targets):

        # print("before output: ", outputs)
        output = self.tree.interpret_batched_prediction_greedy(outputs)[1]
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
    
class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, classes):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.tree = classes
    
    def forward(self, outputs, targets):
        ce_loss = self.ce_loss(outputs, targets)
        batch_size = targets.size(0)
        weights = torch.tensor([lca_distance(formatText(self.tree[pred]), formatText(self.tree[true])) for pred, true in zip(outputs.argmax(dim=1), targets)])
        weighted_loss = ce_loss * weights.to(outputs.device)
        return weighted_loss.mean()
    
class TreeNode:
    def __init__(self, name, fullname, parent = None):
        self.name = fullname
        self.fullname = fullname
        self.children = {}
        self.parent = parent
        self.nick_name = name
    
    def add_child(self, child_name):
        child_fullname = self.fullname + "_" + child_name
        if child_fullname not in self.children:
            self.children[child_fullname] = TreeNode(child_name, child_fullname, self)
        return self.children[child_fullname]

class Tree:
    def __init__(self, classes = None):
        self.root = TreeNode("root", "root")
        self.nodes = {self.root.fullname: self.root}

        # "a_b" => i
        self.edge_to_id = {}

        # i => "a_b"
        self.edges = []

        # drop unessecary number tag
        self.classes = self.transform_classes(classes)
        if (self.classes != None):
            for cls in self.classes:
                # build tree
                self.add_path(cls)

        # bfs and record edges
        self.record_edges()

    def get_edge_parent(self, edge_index):
        (par, child) = self.edges[edge_index]
        parents_parent = par.parent

        if (parents_parent == None):
            return None
        
        edge_key = parents_parent.name + "_" + par.name
        edge_id = self.edge_to_id[edge_key]

        return edge_id
    
    # def diverged(self, edge_index1, edge_index2):
        

    def record_edges(self):
        q = queue.Queue()
        q.put(self.root)


        while (not q.empty()):
            explore = q.get()

            for child in explore.children.values():
                # add edge
                self.edges.append((explore, child))

                edge_key = explore.name + "_" + child.name
                self.edge_to_id[edge_key] = len(self.edges) - 1

                q.put(child)

    def add_path(self, path):
        node_names = path.split('_')
        current_node = self.root

        for node_name in node_names:
            child_node = None
            if node_name not in current_node.children:
                child_node = current_node.add_child(node_name)
                self.nodes[child_node.fullname] = child_node
            else:
                child_node = current_node.children[node_name]
                
            current_node = child_node

    def get_node_by_name(self, node_name):
        return self.nodes.get(node_name, None)


    def print_tree(self, node=None, level=0):
        if node is None:
            self.print_edges()
            node = self.root
        print(" " * level * 2 + f"{node.nick_name}")
        for child in node.children.values():
            self.print_tree(child, level + 1)

    def print_edges(self):
        print(self.edge_to_id)
        for edge in self.edges:
            print(edge[0].nick_name + "_" + edge[1].nick_name)

    def get_target_path(self, path):
        target = [0] * self.get_num_edges()
        # print("input path: ", path)
        node = self.nodes["root_" + path]

        while(node.parent != None):

            parent_name = node.parent.fullname
            child_name = node.fullname

            edge_key = parent_name + "_" + child_name
            edge_index = self.edge_to_id[edge_key]

            target[edge_index] = 1

            node = node.parent
        
        # print("target: ", target)
        result_string = "_".join(result_string.split("_")[1:])
        return target
        
    
    def transform_classes(self, classes):
        transformed = None
        if (classes != None):
            transformed = [formatText(s) for s in classes]
        return transformed

    def get_num_edges(self):
        return len(self.edges)
    
    def interpret_prediction_greedy(self, edge_indicators):
        result = [0] * self.get_num_edges()
        # result_string = ""

        curr_node = self.root

        while (len(curr_node.children) > 0):
            # while we have children
            # make a choice
            max_curr = 0
            max_index = -1

            # print("looking at: ", curr_node.fullname)

            for child_id, child in enumerate(curr_node.children.values()):
                edge_key = curr_node.name + "_" + child.name
                # print("checking child: ", edge_key)
                edge_index = self.edge_to_id[edge_key]

                if (edge_indicators[edge_index] > max_curr):
                    max_curr = edge_indicators[edge_index]
                    max_index = edge_index

            # now we have best index to take
            # print("choosing child at id {} with value {}".format(max_index, max_curr))
            edge_object = self.edges[max_index]
            result[max_index] = 1
            # result_string += edge_object[0].name + "_"

            curr_node = edge_object[1]
            result_string = curr_node.fullname

            result_string = "_".join(result_string.split("_")[1:])

        return result_string, result
        

    def interpret_batched_prediction_greedy(self, edge_indicators):

        predicted_strings = []
        interpreted_batch = []

        for batch_index in range(edge_indicators.shape[0]):

            (pred_string, pred_edges) = self.interpret_prediction_greedy(edge_indicators[batch_index])
            predicted_strings.append(pred_string)
            interpreted_batch.append(pred_edges)

        return predicted_strings, interpreted_batch


    def max_probability_path(self, edge_logits):
        # print("edges: ", self.get_num_edges())
        # print("probs: ", len(edge_probabilities))
        edge_probabilities = self.softmax(edge_logits)
        edge_probabilities = edge_logits


        result = [0] * self.get_num_edges()
        result_string = ""
        dp = [-1] * self.get_num_edges()
        pred = [-1] * self.get_num_edges()
        for i in range(self.get_num_edges()-1, -1, -1):
            v = self.edges[i][0]
            u = self.edges[i][1]
            if (len(u.children) <= 0):
                dp[i] = edge_probabilities[i]
            else:
                my_value = edge_probabilities[i]
                best_val = my_value
                best_par = -1
                for child in u.children.values():
                    edge_key = u.name + "_" + child.name
                    edge_index = self.edge_to_id[edge_key]

                    new_value = my_value + dp[edge_index]
                    if (new_value > best_val):
                        best_val = new_value
                        best_par = edge_index

                dp[i] = best_val
                pred[i] = best_par

        # trace path
        best_succ = -1
        max_val = 0
        for child_id, child in enumerate(self.root.children.values()):
            edge_key = self.root.name + "_" + child.name
            edge_index = self.edge_to_id[edge_key]


            child_path = dp[edge_index]
            if (child_path > max_val):
                max_val = child_path
                best_succ = child_id
            
        
        result[best_succ] = 1
        result_string = self.edges[best_succ][1].fullname
        while(pred[best_succ] != -1):
            best_succ = pred[best_succ]

            #now we have edge id of next one
            result[best_succ] = 1
            result_string = self.edges[best_succ][1].fullname

        # print("result string: ", result_string)
        result_string = "_".join(result_string.split("_")[1:])
        # print("after transform: ", result_string)

        return result_string, result

        
    def softmax(self, edge_logits):
        result = [0] * self.get_num_edges()
        self.hierarchical_softmax(self.root, edge_logits, result)
        return result

    def hierarchical_softmax(self, curr_node, edge_logits, result):

        # softmax this nodes children
        indices = []
        values = []
        for child_id, child in enumerate(curr_node.children.values()):
            edge_key = curr_node.name + "_" + child.name
            # print("checking child: ", edge_key)
            edge_index = self.edge_to_id[edge_key]

            values.append(edge_logits[edge_index])
            indices.append(edge_index)

        values = torch.softmax(torch.tensor(values, dtype=torch.float), dim=0).tolist()
        for i, index in enumerate(indices):
            result[index] = values[i]

        #soft max complete for this node, now do this for every child
        for child_id, child in enumerate(curr_node.children.values()):
            self.hierarchical_softmax(child, edge_logits, result)

    def max_probability_path_batched(self, edge_indicators):
        predicted_strings = []
        interpreted_batch = []

        for batch_index in range(edge_indicators.shape[0]):

            (pred_string, pred_edges) = self.max_probability_path(edge_indicators[batch_index])
            predicted_strings.append(pred_string)
            interpreted_batch.append(pred_edges)

        return predicted_strings, interpreted_batch
"""
"s_a_b_c" => [0, 1, 1, 0, 0, 1] (encoding the edges sa, ab, and bc)
[0, 1, 1, 0, 0, 1] => "s_a_b_c"

[ax, sa, ab, sb, ca, bc]
[0.1, 0.9, 0.2, 0.3, 0.5]
"""

def test_1():
    classes = [
        "1_a_b_c_d",
        "1_a_b_c_e",
        "1_a_b_f_g",
        "1_a_b_f_h",
        "1_a_i_j_k",
        "1_a_i_j_l",
        "1_a_i_m_n",
        "1_a_i_m_o"
    ]

    tree = Tree(classes)
    tree.print_tree()

    # assert tree.get_target_path("a_b_c_d") == [1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0] , print(tree.get_target_path("a_b_c_d"))
    # assert tree.get_target_path("a_b_c_e") == [1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0] , print(tree.get_target_path("a_b_c_e"))
    # assert tree.get_target_path("a_b_f_g") == [1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0] , print(tree.get_target_path("a_b_f_g"))
    # assert tree.get_target_path("a_b_f_h") == [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0] , print(tree.get_target_path("a_b_f_h"))
    # assert tree.get_target_path("a_i_j_k") == [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0] , print(tree.get_target_path("a_i_j_k"))
    # assert tree.get_target_path("a_i_j_l") == [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0] , print(tree.get_target_path("a_i_j_l"))
    # assert tree.get_target_path("a_i_m_n") == [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0] , print(tree.get_target_path("a_i_m_n"))
    # assert tree.get_target_path("a_i_m_o") == [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1] , print(tree.get_target_path("a_i_m_o"))

    # assert tree.interpret_prediction_greedy([1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])[0] == "a_b_c_d"
    # assert tree.interpret_prediction_greedy([1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])[0] == "a_b_c_e"
    # assert tree.interpret_prediction_greedy([1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])[0] == "a_b_f_g"
    # assert tree.interpret_prediction_greedy([1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])[0] == "a_b_f_h"
    # assert tree.interpret_prediction_greedy([1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0])[0] == "a_i_j_k"
    # assert tree.interpret_prediction_greedy([1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0])[0] == "a_i_j_l"
    # assert tree.interpret_prediction_greedy([1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0])[0] == "a_i_m_n"
    # assert tree.interpret_prediction_greedy([1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1])[0] == "a_i_m_o"
    # tree.print_tree()


    # print(tree.get_edge_parent(14))
    # print(tree.get_edge_parent(0))

    # labels = torch.tensor([[0.,  1.,   0.,   0.,    1.,    0.,   0.,  0.,   0.,   0.,  1.,  0.,  0.,  0.],
    #                        [0.,  1.,   0.,   0.,    1.,    0.,   0.,  0.,   0.,   0.,  1.,  0.,  0.,  0.]])
    # output = torch.tensor([[0.1, 0.3, -2.3,  -3.5,  7.3,  -2.1,  0.1, 0.1, -9.1, -2.3, 5.0, 0.2, 1.2, 2.3],
    #                        [0.1, -9, -2.3,  -3.5,  7.3,  -2.1,  0.1, 0.1, -9.1, -2.3, -5.0, 0.2, 1.2, 2.3]])

    # print("labels shape: ", labels.shape)
    # print("output shape: ", output.shape)

    # criterion = LCAPathLoss(classes, clamp_loss=True)
    # loss = criterion(output, labels)
    # print("loss: ", loss)

    # Example usage:
    # classes = ['node0:0_node1:0_node2:0_node3:0', 'node0:0_node1:0_node2:1_node3:0', 'node0:0_node1:1_node2:0_node3:0', 'node0:0_node1:1_node2:1']
    # tree = Tree(classes)
    # tree.print_tree()
    # probabilities = [0.51037731971617, 0.18557036646995217, 0.44019038120086607, 0.25064133340328565, 0.7756077707802558, 0.0015778617201951395, 0.32582208082226405, 0.8486249203636154, 0.7343616583810292]
    probabilities = [0.1, 0.3, -2.3,  -3.5,  7.3,  -2.1,  0.1, 0.1, -9.1, -2.3, 5.0, 0.2, 1.2, 2.3, 20]
    softmaxed_probs = tree.softmax(probabilities)
    print("softmax: ", softmaxed_probs)
    one_hot_encoded_path = tree.max_probability_path(probabilities)
    greedy_path = tree.interpret_prediction_greedy(probabilities)
    print("One-hot encoded path:", one_hot_encoded_path)
    print("Greedy              :", greedy_path)

if (__name__ == "__main__"):
    test_1()