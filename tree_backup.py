import torch
import queue

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
    def __init__(self, classes = None, depth=1):
        self.root = TreeNode("root", "root")
        self.nodes = {self.root.fullname: self.root}
        self.depth = depth
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
        
        return target
    
    def formatText(self, class_label):
        return "_".join(class_label.split("_")[self.depth:])
        
    def get_target_path_batched(self, raw_labels):
        target_strings = []
        target_edges = []

        for batch_index in range(raw_labels.shape[0]):

            curr_string_prediction = self.classes[raw_labels[batch_index]]
            edge_indicators = self.get_target_path(curr_string_prediction)
            
            target_strings.append(curr_string_prediction)
            target_edges.append(edge_indicators)

        return target_strings, torch.tensor(target_edges,  dtype=torch.float32)
    
    def transform_classes(self, classes):
        transformed = None
        if (classes != None):
            transformed = [self.formatText(s) for s in classes]
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
        

    def interpret_prediction_greedy_batched(self, edge_indicators):

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