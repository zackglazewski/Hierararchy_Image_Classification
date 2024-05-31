from torch import nn, optim
import torch
import torchvision
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import random
from tqdm import tqdm
from LCALoss import *
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

def transform_batch_labels_to_edge(tree, target_ids):
    batch_size = len(target_ids)
    edge_indicators = []
    
    for i in range(batch_size):
        target_string = formatText(classes[(target_ids[i])])
        edge_indicator = tree.get_target_path(target_string)
        edge_indicators.append(edge_indicator)
    
    return torch.tensor(edge_indicators, dtype=torch.float32)

def transform_batch_output(tree, output):
    batch_size = len(output)
    paths = []

    for i in range(batch_size):
        out = output[i]
        paths.append(tree.interpret_prediction_greedy(out)[0])

    return paths

def transform_batch_labels_to_string(tree, target_ids):
    batch_size = len(target_ids)
    paths = []
    for i in range(batch_size):
        target_string = formatText(classes[(target_ids[i])])
        paths.append(target_string)

    return paths

def get_classes(data_dir):
    all_data = datasets.ImageFolder(data_dir)
    return all_data.classes, len(all_data)

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
        # train_data_len = int(len(all_data))
        valid_data_len = int((len(all_data) - train_data_len)/2)
        test_data_len = int(len(all_data) - train_data_len - valid_data_len)
        train_data, val_data, test_data = random_split(all_data, [train_data_len, valid_data_len, test_data_len])
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
        # train_loader = DataLoader(all_data, batch_size=train_data_len, shuffle=False, num_workers=0)
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
    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

data_dir = "inat2021birds/bird_train"
(train_loader, train_size) = get_data_loaders(data_dir, batch_size=32, train=True)
(val_loader, test_loader, valid_data_len, test_data_len) = get_data_loaders(data_dir, batch_size=32, train=False)

(classes, num_instances) = get_classes(data_dir)
print(classes)

taxonomy = Tree(classes)
taxonomy.print_tree()

# labels = y = torch.tensor([[1., 1., 0.],[1., 0., 1.]])

model = timm.create_model("hf_hub:timm/efficientnet_b3.ra2_in1k", pretrained=True)
for param in model.parameters():
    param.requires_grad = False
n_inputs = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Linear(n_inputs,2048),
    nn.SiLU(),
    nn.Dropout(0.3),
    nn.Linear(2048, taxonomy.get_num_edges())
)
model = model.to(device)
# print(model)
criterion = nn.BCEWithLogitsLoss()
criterion = criterion.to(device)
optimizer = optim.Adam(model.classifier.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.97)

best_validation_accuracy = 0

for i in range(100):
    print("Training Epoch {}".format(i))
    running_correct = 0
    running_loss = 0
    model.train()
    for inputs, labels in tqdm(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(inputs)

        transformed_labels = transform_batch_labels_to_edge(taxonomy, labels)
        transformed_labels = transformed_labels.to(device)

        loss = criterion(output, transformed_labels)

        
        loss.backward()
        optimizer.step()

        # print("output: ", output)
        # print("label: ", transformed_labels)
        # print('Loss: {:.3f}'.format(loss.item()))

        path_preds = transform_batch_output(taxonomy, output)
        path_labels = transform_batch_labels_to_string(taxonomy, labels)

        # print("path_preds : ", path_preds)
        # print("path_labels: ", path_labels)
        # print("len: ", len(dataset))
        running_correct += count_equal(path_preds, path_labels)
        running_loss += loss
    # scheduler.step()
    print("train loss\t{}\ttrain accuracy\t{}".format(running_loss / train_size, running_correct / train_size))

    print("Validation Epoch {}".format(i))
    running_correct = 0
    running_loss = 0
    model.eval()
    for inputs, labels in tqdm(val_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        with torch.set_grad_enabled(False):
            output = model(inputs)

            transformed_labels = transform_batch_labels_to_edge(taxonomy, labels)
            transformed_labels = transformed_labels.to(device)

            loss = criterion(output, transformed_labels)

        

            # print("output: ", output)
            # print("label: ", transformed_labels)
            # print('Loss: {:.3f}'.format(loss.item()))

            path_preds = transform_batch_output(taxonomy, output)
            path_labels = transform_batch_labels_to_string(taxonomy, labels)

        # print("path_preds : ", path_preds)
        # print("path_labels: ", path_labels)
        # print("len: ", len(dataset))
        running_correct += count_equal(path_preds, path_labels)
        running_loss += loss
    # scheduler.step()
    val_accuracy = running_correct / valid_data_len
    if (val_accuracy > best_validation_accuracy):
        best_validation_accuracy = val_accuracy
        torch.save(model.state_dict(), "best.pth")

    print("val loss\t{}\tval accuracy\t{}".format(running_loss / valid_data_len, val_accuracy))
    print()

torch.save(model.state_dict(), "last.pth")

print("Testing With Best")
model.load_state_dict(torch.load("best.pth"))
model.eval()

running_correct = 0
running_loss = 0
for inputs, labels in tqdm(test_loader):
    inputs = inputs.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()

    with torch.set_grad_enabled(False):
        output = model(inputs)

        transformed_labels = transform_batch_labels_to_edge(taxonomy, labels)
        transformed_labels = transformed_labels.to(device)

        loss = criterion(output, transformed_labels)


        path_preds = transform_batch_output(taxonomy, output)
        path_labels = transform_batch_labels_to_string(taxonomy, labels)

    running_correct += count_equal(path_preds, path_labels)
    running_loss += loss


print("Final Test loss\t{}\tFinal Test accuracy\t{}".format(running_loss / valid_data_len, running_correct/test_data_len))
print()

print("Testing With Last")
model.load_state_dict(torch.load("last.pth"))
model.eval()

running_correct = 0
running_loss = 0
for inputs, labels in tqdm(test_loader):
    inputs = inputs.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()

    with torch.set_grad_enabled(False):
        output = model(inputs)

        transformed_labels = transform_batch_labels_to_edge(taxonomy, labels)
        transformed_labels = transformed_labels.to(device)

        loss = criterion(output, transformed_labels)


        path_preds = transform_batch_output(taxonomy, output)
        path_labels = transform_batch_labels_to_string(taxonomy, labels)

    running_correct += count_equal(path_preds, path_labels)
    running_loss += loss


print("Final Test loss\t{}\tFinal Test accuracy\t{}".format(running_loss / valid_data_len, running_correct/test_data_len))
print()