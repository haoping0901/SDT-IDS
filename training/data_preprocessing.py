import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from typing import Dict

def make_dataloader(config: dict) -> Dict[data.DataLoader, 
                                          data.DataLoader]:
    # Load dataset into dataloader
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=config["channel"]),
        transforms.Resize((config["image_size"], 
                           config["image_size"])), 
        transforms.ToTensor()])
    
    # match config["data_set"]:
    #     case "binary_match":
    #         datapath = BINARY_MATCH
    #     case "multi_match":
    #         datapath = MULTI_MATCH

    data_loader = dict()
    trainset = torchvision.datasets.ImageFolder(
        # root=datapath["train"], transform=transform)
        root=config["data_path"] + "/train/", transform=transform)
    data_loader["train"] = data.DataLoader(
        trainset, batch_size=config["batch_size"], shuffle=True, 
        num_workers=4)
    testset = torchvision.datasets.ImageFolder(
        # root=datapath["test"], transform=transform)
        root=config["data_path"] + "/test/", transform=transform)
    data_loader["test"]  = data.DataLoader(
        testset, batch_size=config["batch_size"], shuffle=True, 
        num_workers=4)
    
    # print(f"Dataset using: {datapath}")
    # print(f"Train label: {trainset.class_to_idx}")
    # print(f"Test label: {testset.class_to_idx}")

    return data_loader

# class 