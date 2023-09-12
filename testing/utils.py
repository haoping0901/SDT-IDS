import torch
import torch.utils.data as data
from typing import Any
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time

def _plot_cfm(cfm: np.array, model_dir: str) -> None:
    print(f"cfm: {cfm}")

    fig, ax = plt.subplots(1, 1)
    plt.title("Confusion Matrix")
    cfm_fig = ax.imshow(cfm)
    
    plt.xlabel("Label")
    plt.ylabel("Prediction")

    x_label_list = ["Benign", "Malicious"]
    y_label_list = ["Malicious", "Benign"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(x_label_list)
    ax.set_yticklabels(y_label_list)

    # Set the number for each coordinate.
    # Be cautious that the order of text function is reversed to the 
    # array.
    for i in range(2):
        for j in range(2):
            if i != j:
                ax.text(j, i, str(int(cfm[i][j])), ha="center", 
                           va="center", color="w", fontsize="large", 
                           fontweight="bold")
            else:
                ax.text(j, i, str(int(cfm[i][j])), ha="center", 
                           va="center", color="b", fontsize="large", 
                           fontweight="bold")
    
    fig.colorbar(cfm_fig)
    
    plt.savefig(f"{model_dir}/cfm.png")
    plt.close()

def evaluation(
        model: torch.nn.Module, data_loader: data.DataLoader, 
        criterion: Any, device: torch.device, model_dir: str
        ) -> None:
    model.to(device)
    model.eval()
    corrects = 0
    test_loss = 0.0

    cfm_shape = (len(data_loader["test"].dataset.class_to_idx), 
                 len(data_loader["test"].dataset.class_to_idx))
    
    # cfm [[TN, FN]
    #      [FP, TP]]
    cfm = np.zeros(cfm_shape)

    with torch.no_grad():
        for instances, labels in tqdm(data_loader["test"]):
            instances = instances.to(device)
            labels = labels.to(device)

            outputs = model(instances)

            # Accumulate the correct predictions.
            _, preds = torch.max(outputs, dim=1)
            corrects += torch.sum(preds == labels.data)

            # make confusion matrix
            for row, col in zip(preds, labels.data):
                cfm[row, col] += 1
            # Obtain the loss
            loss = criterion(outputs, labels)
            test_loss += loss
    
    data_loader_len = len(data_loader["test"])
    test_instances_num = (data_loader_len 
                          * data_loader["test"].batch_size)
    test_loss /= data_loader_len
    test_accuracy = (corrects / test_instances_num) * 100

    TN = cfm[0, 0]; FN = cfm[0, 1]; FP = cfm[1, 0]; TP = cfm[1, 1]
    print(f"\nTN: {TN}")
    print(f"FN: {FN}")
    print(f"FP: {FP}")
    print(f"TP: {TP}\n")
    precision = TP / (TP+FP)
    recall = TP / (TP+FN)
    f1_score = 2*precision*recall / (precision+recall)
    with open(f"{model_dir}/evaluation.txt", mode='w') as file:
        file.write(f"Accuracy: {test_accuracy:.2f} %\n")
        file.write(f"Precision: {(precision*100):.2f} %\n")
        file.write(f"Recall: {(recall*100):.2f} %\n")
        file.write(f"F1-score: {(f1_score*100):.2f} %\n")
    
    _plot_cfm(cfm, model_dir)
    print(f"\nAccuracy: {test_accuracy:.2f} %")
    print(f"Precision: {(precision*100):.2f} %")
    print(f"Recall: {(recall*100):.2f} %")
    print(f"F1-score: {(f1_score*100):.2f} %\n")
    
    print(72*'*')
    print('*' + 70*' ' + '*')
    print(f"*\tAccuracy: {test_accuracy:.2f} %" + 46*' ' + '*')
    print(f"*\tTesting loss: {test_loss:.4f}" + 43*' ' + '*')
    print('*' + 70*' ' + '*')
    print(72*'*')

    # return cfm

def response_time(model: torch.nn.Module, model_dir: str, 
                  data_loader: data.DataLoader, device: torch.device
                  ) -> None:
    model.to(device)
    model.eval()

    with torch.no_grad():
        start_time = time.time()
        # for instances, labels in tqdm(data_loader["test"]):
        for instances, labels in data_loader["test"]:
            instances = instances.to(device)
            labels = labels.to(device)

            outputs = model(instances)
            _, preds = torch.max(outputs, dim=1)    
        time_elapsed = time.time() - start_time

    test_instances_num = (len(data_loader["test"]) 
                          * data_loader["test"].batch_size)
    time_ms = (time_elapsed / test_instances_num) * 1000

    with open(f"{model_dir}/time_elapsed.txt", mode='a') as file:
        file.write(f"Time elapsed: {time_ms:.4f} ms\n")

    print(72*'*')
    print('*' + 70*' ' + '*')
    print(f"*\tTime elapsed: {time_ms:2f} ms" + 38*' ' + '*')
    print('*' + 70*' ' + '*')
    print(72*'*')