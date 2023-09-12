import torch
from training import train, data_preprocessing
import os
from parser.parser import set_model
import numpy as np

# np.random.seed(42)
torch.manual_seed(42)

if __name__ == "__main__":
    # Set the model from command line arguments
    config = set_model.main(standalone_mode=False)
    if config == 0:
        exit()
    # print(config)
    
    cur_dir = os.path.dirname(__file__)

    # Prepare the data loader
    # class to idx: {'benign': 0, 'malicious': 1}
    data_loader = data_preprocessing.make_dataloader(config=config)

    # Prepare the training configurations of the model
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(config["model"].parameters())
    # optimizer = torch.optim.SGD(config["model"].parameters(), 
    #                             lr=0.01, momentum=0.9)

    # Start training
    train.train(
        data_loader, config["model"], criterion, optimizer, 
        config["data_set"], epochs=config["epoch"], 
        device=config["device"], cur_dir=cur_dir, 
        model_name=config["model_name"], 
        model_config=config["model_config"])