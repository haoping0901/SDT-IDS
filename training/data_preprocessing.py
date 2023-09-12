import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from typing import Dict

SUBSET_MERGED = {
    "train": "/home/EA301B/611410037/CICDDoS2019/IGTD_Conversion/" 
             + "subset/train_merged", 
    "test": "/home/EA301B/611410037/CICDDoS2019/IGTD_Conversion/" 
            + "subset/test_merged"}
SUBSET_MERGED_ALL = {
    "train": ("/home/EA301B/611410037/CICDDoS2019/" 
              + "IGTD_Conversion/IGTD_all/train"), 
    "test": ("/home/EA301B/611410037/CICDDoS2019/" 
             + "IGTD_Conversion//IGTD_all/test")}
SUBSET_MERGED_ALL_SPLIT = {
    "train": "/home/EA301B/611410037/CICDDoS2019/IGTD_Conversion/" 
             + "IGTD_all_split/train", 
    "test": "/home/EA301B/611410037/CICDDoS2019/IGTD_Conversion/" 
            + "IGTD_all_split/test"}
SUBEST_BINARY_MATCH = {
    "train": ("/home/EA301B/611410037/CICDDoS2019/IGTD_Conversion/" 
              + "binary_match/train"), 
    "test": ("/home/EA301B/611410037/CICDDoS2019/IGTD_Conversion/" 
             + "binary_match/test")
}
SUBSET_MERGED_ALL_MATCH = {
    "train": "/home/EA301B/611410037/CICDDoS2019/IGTD_Conversion/" 
             + "IGTD_all_match/train", 
    "test": "/home/EA301B/611410037/CICDDoS2019/IGTD_Conversion/" 
            + "IGTD_all_match/test"}
SUBSET_DIVIDED = {
    "train": "/home/EA301B/611410037/CICDDoS2019/IGTD_Conversion/" 
             + "subset/train_divided", 
    "test": "/home/EA301B/611410037/CICDDoS2019/IGTD_Conversion/" 
            + "subset/test_divided"}
SUBSET_DIVIDED_MATCH = {
    "train": "/home/EA301B/611410037/CICDDoS2019/IGTD_Conversion/" 
             + "subset_match/train", 
    "test": "/home/EA301B/611410037/CICDDoS2019/IGTD_Conversion/" 
            + "subset_match/test"}

def make_dataloader(config: dict) -> Dict[data.DataLoader, 
                                          data.DataLoader]:
    # Load dataset into dataloader
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=config["channel"]),
        transforms.Resize((config["image_size"], 
                           config["image_size"])), 
        transforms.ToTensor()])
    
    match config["data_set"]:
        case "merged":
            datapath = SUBSET_MERGED
        case "merged_all":
            datapath = SUBSET_MERGED_ALL
        case "binary_match":
            datapath = SUBEST_BINARY_MATCH
        case "merged_all_match":
            datapath = SUBSET_MERGED_ALL_MATCH
        case "merged_all_split":
            datapath = SUBSET_MERGED_ALL_SPLIT
        case "divided":
            datapath = SUBSET_DIVIDED
        case "divided_match":
            datapath = SUBSET_DIVIDED_MATCH

    data_loader = dict()
    trainset = torchvision.datasets.ImageFolder(
        root=datapath["train"], transform=transform)
    data_loader["train"] = data.DataLoader(
        trainset, batch_size=config["batch_size"], shuffle=True, 
        num_workers=4)
    testset = torchvision.datasets.ImageFolder(
        root=datapath["test"], transform=transform)
    data_loader["test"]  = data.DataLoader(
        testset, batch_size=config["batch_size"], shuffle=True, 
        num_workers=4)
    
    # print(f"Dataset using: {datapath}")
    # print(f"Train label: {trainset.class_to_idx}")
    # print(f"Test label: {testset.class_to_idx}")

    return data_loader

# class 