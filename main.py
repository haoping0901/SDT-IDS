import torch
from training import training, data_preprocessing
import os
from parser.train_parser import train

torch.manual_seed(42)

if __name__ == "__main__":
    # Set the model from command line arguments
    config = train(standalone_mode=False)
    if config == 0:
        exit()
    
    # cur_dir = os.path.dirname(__file__).rsplit('/', maxsplit=1)[0]
    cur_dir = os.path.dirname(__file__)
    # Prepare the data loader
    data_loader = data_preprocessing.make_dataloader(config=config)

    # Prepare the training configurations of the model
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(config["model"].parameters())

    # Start training
    training.train(
        data_loader, config["model"], criterion, optimizer, 
        epochs=config["epoch"], device=config["device"], 
        cur_dir=cur_dir, model_name=config["model_name"], 
        model_config=config["model_config"])