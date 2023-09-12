import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

SUBSET_MERGED_ALL = {
    "train": "/home/EA301B/611410037/CICDDoS2019/IGTD_Conversion/" 
             + "IGTD_all/train", 
    "test": "/home/EA301B/611410037/CICDDoS2019/IGTD_Conversion/" 
            + "IGTD_all/test"}

data_loader = dict()
trainset = torchvision.datasets.ImageFolder(
    root=SUBSET_MERGED_ALL["train"])
data_loader["train"] = data.DataLoader(
    trainset, batch_size=64, shuffle=True, 
    num_workers=2)
testset = torchvision.datasets.ImageFolder(
    root=SUBSET_MERGED_ALL["test"])
data_loader["test"]  = data.DataLoader(
    testset, batch_size=64, shuffle=True, 
    num_workers=2)

print(f"classes: {trainset.classes}")
print(f"class to idx: {trainset.class_to_idx}")

print(f"classes: {testset.classes}")
print(f"class to idx: {testset.class_to_idx}")