import torch
from torchinfo import summary
from training.data_preprocessing import make_dataloader
import vit_pytorch
from typing import Any
from testing import utils
from testing import lnn, dcnn, xgbfs_cnn_lstm

torch.manual_seed(42)

EVAL_MODE = {0: "performance", 1: "time_elapsed", 2: "model_summary"}
EVAL_IDX = 1
# DEVICE = 3
DEVICE = "cpu"
MODEL_PATH = ("path_to_model")

config = {
    "model": "SimpleViT", 
    "image_size": 4,
    "patch_size": 2,
    "channel": 1, 
    "num_classes": 2,
    "dim": 8,
    "mlp_dim": 8,
    "depth": 1, 
    "heads": 1, 
    }
config["data_set"] = "binary_match"
config["batch_size"] = 64

def check_device(idx: int, model: Any) -> None:
    print(f"Check {idx}:")
    for layer in model.named_parameters():
        print(f"{layer[0]} -> {layer[1].device}")


if __name__ == "__main__":
    device = torch.device(DEVICE)
    data_loader = make_dataloader(config)
    
    # Set model
    match config["model"]:
        case "SimpleViT":
            model = vit_pytorch.SimpleViT(
                image_size=config["image_size"], 
                patch_size=config["patch_size"], 
                num_classes=config["num_classes"], 
                dim=config["dim"], 
                depth=config["depth"], 
                heads=config["heads"], 
                mlp_dim=config["mlp_dim"], 
                channels=config["channel"])
        case "LNN":
            model = lnn.LNN(channel=config["channel"], 
                            n_class=config["num_classes"])
        case "DCNN":
            model = dcnn.DCNN(channel=config["channel"], 
                              n_class=config["num_classes"])
        case "CNN_LSTM":
            model = xgbfs_cnn_lstm.CNN_LSTM(
                channel=config["channel"], 
                n_classes=config["num_classes"], 
                features=config["image_size"]**2)

    # check_device(0, model)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    # check_device(1, model)

    criterion = torch.nn.CrossEntropyLoss()
    
    match EVAL_MODE[EVAL_IDX]:
        case "performance":
            utils.evaluation(model, data_loader, criterion, device, 
                             MODEL_PATH.rsplit('/', 1)[0])
        case "time_elapsed":
            utils.response_time(model, MODEL_PATH.rsplit('/', 1)[0], 
                                data_loader, device)
        case "model_summary":
            # Show model
            summary(model=model, 
                    input_size=(config["batch_size"], 
                                config["channel"], 
                                config["image_size"], 
                                config["image_size"]), 
                    device=device, show_unit=1)