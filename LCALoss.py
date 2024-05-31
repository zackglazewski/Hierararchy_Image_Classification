from torch import nn, sigmoid
import torch
import queue
import copy
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
    return "_".join(class_label.split("_")[4:])
    # return "_".join(class_label.split("_")[6:])
    # return "_".join(class_label.split("_")[1:])


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
        self.name = name
        self.fullname = fullname
        self.children = {}
        self.parent = parent
    
    def add_child(self, child_name):
        if child_name not in self.children:
            child_fullname = self.fullname + "_" + child_name
            self.children[child_name] = TreeNode(child_name, child_fullname, self)
        return self.children[child_name]

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
        print(" " * level * 2 + f"{node.name}")
        for child in node.children.values():
            self.print_tree(child, level + 1)

    def print_edges(self):
        print(self.edge_to_id)
        for edge in self.edges:
            print(edge[0].name + "_" + edge[1].name)

    def get_target_path(self, path):
        target = listofzeros = [0] * len(self.edges)

        node_names = path.split("_")
        assert len(node_names) > 1 , "tree does not have any edges"

        # connect root
        edge_key = "root" + "_" + node_names[0]
        edge_index = self.edge_to_id[edge_key]
        target[edge_index] = 1

        for i in range(len(node_names)-1):
            node1 = node_names[i]
            node2 = node_names[i+1]

            edge_key = node1 + "_" + node2
            edge_index = self.edge_to_id[edge_key]

            target[edge_index] = 1

        return target
    
    def transform_classes(self, classes):
        transformed = None
        if (classes != None):
            transformed = [formatText(s) for s in classes]
        return transformed

    def get_num_edges(self):
        return len(self.edges)
    
    def interpret_prediction_greedy(self, edge_indicators):
        predicted_path = ""
        onehot_edges = [0] * len(edge_indicators)
        
        # inclusive
        start = 0
        current_node = self.root

        while (start < len(edge_indicators)):
            # start and current node is already set and is inclusive

            # exclusive
            end = start + len(current_node.children)
            options = torch.Tensor(edge_indicators[start:end])
            chosen_edge_index = start + torch.argmax(options)
            onehot_edges[chosen_edge_index] = 1
            chosen_edge = self.edges[chosen_edge_index]
            predicted_path += chosen_edge[1].name + "_"

            current_node = chosen_edge[1]

            if (len(current_node.children) <= 0):
                break
            else:
                first_child_name = ""
                for child_key in current_node.children.keys():
                    first_child_name = child_key
                    break
                edge_key = current_node.name + "_" + first_child_name
                start = self.edge_to_id[edge_key]


        return predicted_path[0:-1], onehot_edges

    def interpret_batched_prediction_greedy(self, edge_indicators):

        predicted_strings = []
        interpreted_batch = []

        for batch_index in range(edge_indicators.shape[0]):

            (pred_string, pred_edges) = self.interpret_prediction_greedy(edge_indicators[batch_index])
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
    tree.print_edges()

    assert tree.get_target_path("a_b_c_d") == [1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0] , print(tree.get_target_path("a_b_c_d"))
    assert tree.get_target_path("a_b_c_e") == [1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0] , print(tree.get_target_path("a_b_c_e"))
    assert tree.get_target_path("a_b_f_g") == [1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0] , print(tree.get_target_path("a_b_f_g"))
    assert tree.get_target_path("a_b_f_h") == [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0] , print(tree.get_target_path("a_b_f_h"))
    assert tree.get_target_path("a_i_j_k") == [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0] , print(tree.get_target_path("a_i_j_k"))
    assert tree.get_target_path("a_i_j_l") == [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0] , print(tree.get_target_path("a_i_j_l"))
    assert tree.get_target_path("a_i_m_n") == [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0] , print(tree.get_target_path("a_i_m_n"))
    assert tree.get_target_path("a_i_m_o") == [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1] , print(tree.get_target_path("a_i_m_o"))

    assert tree.interpret_prediction_greedy([1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])[0] == "a_b_c_d"
    assert tree.interpret_prediction_greedy([1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])[0] == "a_b_c_e"
    assert tree.interpret_prediction_greedy([1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])[0] == "a_b_f_g"
    assert tree.interpret_prediction_greedy([1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])[0] == "a_b_f_h"
    assert tree.interpret_prediction_greedy([1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0])[0] == "a_i_j_k"
    assert tree.interpret_prediction_greedy([1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0])[0] == "a_i_j_l"
    assert tree.interpret_prediction_greedy([1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0])[0] == "a_i_m_n"
    assert tree.interpret_prediction_greedy([1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1])[0] == "a_i_m_o"
    tree.print_tree()


    print(tree.get_edge_parent(14))
    print(tree.get_edge_parent(0))

    labels = torch.tensor([[0.,  1.,   0.,   0.,    1.,    0.,   0.,  0.,   0.,   0.,  1.,  0.,  0.,  0.],
                           [0.,  1.,   0.,   0.,    1.,    0.,   0.,  0.,   0.,   0.,  1.,  0.,  0.,  0.]])
    output = torch.tensor([[0.1, 0.3, -2.3,  -3.5,  7.3,  -2.1,  0.1, 0.1, -9.1, -2.3, 5.0, 0.2, 1.2, 2.3],
                           [0.1, -9, -2.3,  -3.5,  7.3,  -2.1,  0.1, 0.1, -9.1, -2.3, -5.0, 0.2, 1.2, 2.3]])

    print("labels shape: ", labels.shape)
    print("output shape: ", output.shape)

    criterion = LCAPathLoss(classes, clamp_loss=True)
    loss = criterion(output, labels)
    print("loss: ", loss)

if (__name__ == "__main__"):
    test_1()