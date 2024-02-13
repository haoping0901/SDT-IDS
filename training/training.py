import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any
from tqdm import tqdm
import numpy as np
import os
from datetime import timedelta, datetime
import copy


UTC_OFFSET = 8
# def print_path():
#     print(f"train path: {__file__}")

def get_time() -> str:
    now = datetime.now() + timedelta(hours=UTC_OFFSET)
    cur_time = now.strftime("%Y%m%d_%H%M%S")
    # print("date and time:",cur_time)

    return cur_time

def train(
        dataloader: Dict[data.DataLoader, data.DataLoader], 
        model: nn.Module, criterion: Any, optimizer: Any, epochs: int, 
        device: torch.device, cur_dir: str = "", model_name: str = "", 
        model_config: str = "") -> None:
    best_loss = np.inf
    best_epoch = 0
    best_accuracy = 0.0

    # setup a tensorboard writer
    dump_path = (cur_dir + "/checkpoints/" + model_name + '/' 
                 + model_config + f"_{get_time()}/")
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)
    tb_writer = SummaryWriter(dump_path)

    # start training
    for epoch in range(epochs):
        print(f"EPOCH {epoch+1}")
        print(f"\n\tTraining Phase\n")
        
        model.train()
        model.to(device)

        idx = 0
        corrects = 0
        train_loss = 0.0

        # Training phase
        for instances, labels in tqdm(dataloader["train"]):
            instances, labels = instances.to(device), labels.to(device)
            idx += 1

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            outputs = model(instances)
            
            # softmax_outputs = softmax_func(outputs)
            # Accumulate the correct predictions.
            _, preds = torch.max(outputs, dim=1)
            corrects += torch.sum(preds == labels.data)

            # Obtain the loss and its gradients.
            loss = criterion(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            train_loss += loss.item()
            if idx%1000 == 0:
                batch_loss = loss.item() / 1000 # loss per batch
                print(f"\n\tbatch {idx} loss: {batch_loss}")
                batch_accuracy = (
                    (corrects 
                     / (idx * dataloader['train'].batch_size)) 
                     * 100)
                print(f"\tbatch {idx} accuracy: {batch_accuracy:.2f} %")

        train_loss /= len(dataloader["train"])

        # Testing phase
        print(f"\n\tTesting Phase\n")
        model.eval()
        
        corrects = 0
        test_loss = 0.0

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for instances, labels in tqdm(dataloader["test"]):
                instances = instances.to(device)
                labels = labels.to(device)

                outputs = model(instances)

                # Accumulate the correct predictions.
                _, preds = torch.max(outputs, dim=1)
                corrects += torch.sum(preds == labels.data)

                # Obtain the loss
                loss = criterion(outputs, labels)
                test_loss += loss

        test_loss /= len(dataloader["test"])
        test_accuracy = (
            (corrects 
             / (len(dataloader["test"]) * dataloader['test'].batch_size)) 
             * 100)
        print(f"\n\tTesting loss: {test_loss:.4f}, " 
              + f"accuracy: {test_accuracy:.2f} %")
        
        tb_writer.add_scalar("Training Loss", train_loss, epoch+1)
        tb_writer.add_scalar("Testing Loss", test_loss, epoch+1)
        tb_writer.add_scalar("Testing Accuracy", test_accuracy, epoch+1)
        tb_writer.flush()
        
        # Track best performance, and save the model's state
        if test_loss < best_loss:
            best_state_dict = copy.deepcopy(model.state_dict())
            best_loss = test_loss
            best_epoch = epoch + 1
            best_accuracy = test_accuracy
        
        print(f"\tBest epoch: {best_epoch}, Best loss: " 
              + f"{best_loss:.4f}, Best accuracy: {best_accuracy:.2f}" 
              + "\n")
    
    torch.save(
        best_state_dict, 
        (f"{dump_path}{model_name}_BE{best_epoch}_BL{best_loss:.4f}" 
         + f"_BA{best_accuracy:.2f}.pt"))
    
    print('\n' + 55*'*')
    print(f"*" + 53*' ' + '*')
    print(f"*  BEST MODEL in EPOCH {best_epoch:<31}*")
    print(f"*  LOSS {best_loss:.4f} ACCURACY {best_accuracy:.2f} %" 
          + 23*' ' + '*')
    print(f"*" + 53*' ' + '*')
    print(55*'*' + '\n')