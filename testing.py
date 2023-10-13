import torch
from torchinfo import summary
from training.data_preprocessing import make_dataloader
from typing import Any
from testing import utils
from parser.test_parser import test

torch.manual_seed(42)

EVAL_MODE = {0: "performance", 1: "time_elapsed", 2: "model_summary"}

def check_device(idx: int, model: Any) -> None:
    print(f"Check {idx}:")
    for layer in model.named_parameters():
        print(f"{layer[0]} -> {layer[1].device}")

if __name__ == "__main__":
    config = test(standalone_mode=False)
    if config == 0:
        exit()
    print(f"config:\n{config}")

    # exit()
    data_loader = make_dataloader(config)

    # Set model
    config["model"].load_state_dict(
        torch.load(config["test_model_path"], 
                   map_location=config["device"])
    )
    criterion = torch.nn.CrossEntropyLoss()
    
    match EVAL_MODE[config["test_mode"]]:
        case "performance":
            utils.evaluation(
                config["model"], data_loader, criterion, 
                config["device"], 
                config["test_model_path"].rsplit('/', 1)[0])
        case "time_elapsed":
            for cnt in range(config["eval_times"]):
                utils.response_time(
                    config["model"], 
                    config["test_model_path"].rsplit('/', 1)[0], 
                    data_loader, config["device"])
        case "model_summary":
            # Show model
            summary(
                model=config["model"], 
                input_size=(config["batch_size"], 
                            config["channel"], 
                            config["image_size"], 
                            config["image_size"]), 
                device=config["device"])