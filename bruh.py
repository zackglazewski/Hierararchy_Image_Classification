import torch
my_list=[1, 2, 3, 4]
def transform(l):
    for i in range(len(l)):
        l[i] += 1

    return

print(my_list)
transform(my_list)
print(my_list)

my_list = torch.softmax(torch.tensor(my_list, dtype=torch.float), dim=0)
print(my_list)
print(my_list.tolist())