import click
import vit_pytorch
import torch
import sys
sys.path.insert(1, "../testing")
from testing import lnn, dcnn, xgbfs_cnn_lstm

config = {}
HYPERPARAMETERS = {}

@click.command()
@click.option(
    "-m", "--model", default="SimpleViT", show_default=True, 
    type=str, help="specify the model to be imported")
@click.option(
    "-ds", "--data-set", default="binary_match", show_default=True, 
    type=click.STRING, help="specify the dataset to be used")
@click.option(
    "-nc", "--num-classes", type=click.IntRange(0, min_open=True), 
    required=True, help="specify the number of classes")
@click.option(
    "-is", "--image-size", default=4, show_default=True, 
    type=click.IntRange(0, min_open=True, max_open=True), 
    help="specify the image size")
@click.option(
    "-ps", "--patch-size", default=2, show_default=True, 
    type=click.IntRange(0, min_open=True, max_open=True), 
    help="specify the patch size")
@click.option(
    "-ch", "--channel", default=3, show_default=True, type=click.INT, 
    help="specify the number of image channel")
@click.option(
    "-dim", default=8, show_default=True, 
    type=click.IntRange(0, min_open=True, max_open=True), 
    help="specify the last dimension of output tensor after linear "
        + "transformation")
@click.option(
    "-md", "--mlp-dim", default=8, show_default=True, type=int, 
    help="specify the dimension of the feed forward layer")
@click.option(
    "-dp", "--depth", default=1, show_default=True, 
    type=click.IntRange(0, min_open=True, max_open=True), 
    help="specify the number of Transformer blocks")
@click.option(
    "-h", "--heads", default=1, show_default=True, 
    type=click.IntRange(0, min_open=True, max_open=True), 
    help="specify the number of heads")
@click.option(
    "-do", "--dropout", default=0.1, show_default=True, 
    type=click.FloatRange(0, 1, max_open=True), 
    help="specify the dropout rate")
@click.option(
    "-edo", "--emb-dropout", default=0.1, show_default=True, 
    type=click.FloatRange(0, 1, max_open=True), 
    help="specify the dropout rate while embedding")
@click.option(
    "--cuda-id", type=click.INT, default=-1, show_default=True, 
    help="specify the cuda device id, -1 for cpu device")
@click.option(
    "--batch-size", default=64, show_default=True, type=click.INT, 
    help="specify the training batch size")
@click.option(
    "-e", "--epoch", default=100, show_default=True, type=click.INT, 
    help="specify the training epoch")
@click.option(
    "--continue-train", default=False, show_default=True, 
    type=click.BOOL, help="continue training: load the latest model")
@click.option(
    "--saved-model-path", type=click.STRING, 
    help="specify the saved model path")
def set_model(model, image_size, patch_size, channel, num_classes, 
              dim, depth, heads, mlp_dim, dropout, emb_dropout, 
              cuda_id, batch_size, epoch, data_set, continue_train, 
              saved_model_path) -> dict:
    """Sepcify the model and its hyperparameters."""
    config["image_size"] = image_size
    config["channel"] = channel
    if cuda_id == -1:
        config["device"] = torch.device("cpu")
    else:
        config["device"] = torch.device(cuda_id)
    config["batch_size"] = batch_size
    config["epoch"] = epoch
    config["data_set"] = data_set
    HYPERPARAMETERS["image_size"] = image_size
    HYPERPARAMETERS["patch_size"] = patch_size
    HYPERPARAMETERS["channel"] = channel
    HYPERPARAMETERS["num_classes"] = num_classes
    HYPERPARAMETERS["dim"] = dim
    HYPERPARAMETERS["mlp_dim"] = mlp_dim
    HYPERPARAMETERS["depth"] = depth
    HYPERPARAMETERS["heads"] = heads
    HYPERPARAMETERS["batch_size"] = batch_size
    HYPERPARAMETERS["epoch"] = epoch

    continue_epoch = 0

    if model == "SimpleViT":
        config["model"] = vit_pytorch.SimpleViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            channels=channel, 
            mlp_dim=mlp_dim
        )
        config["model_name"] = "SimpleViT"
        config["model_config"] = (
            "SimpleViT_is" + str(image_size) + "_ps" + str(patch_size) 
            + "_ch" + str(channel) + "_nc" + str(num_classes) + "_dim" 
            + str(dim) + "_md" + str(mlp_dim) + "_dp" + str(depth) 
            + "_h" + str(heads) + "_bs" + str(batch_size))
    elif model == "ViT":
        HYPERPARAMETERS["dropout"] = dropout
        HYPERPARAMETERS["emb_dropout"] = emb_dropout

        config["model"] = vit_pytorch.ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout
        )
        config["model_name"] = "ViT"
        config["model_config"] = (
            "ViT_is" + str(image_size) + "_ps" + str(patch_size) 
            + "_ch" + str(channel) + "_nc" + str(num_classes) + "_dim" 
            + str(dim) + "_md" + str(mlp_dim) + "_dp" + str(depth) 
            + "_h" + str(heads) + "_do" + str(dropout) + "_edo" 
            + str(emb_dropout) + "_bs" + str(batch_size))
    elif model == "lnn":
        config["model"] = lnn.LNN(channel=channel, n_class=num_classes)
        config["model_name"] = "LNN"
        config["model_config"] = (
            "LNN_is" + str(image_size) + "_ch" + str(channel) + "_nc" \
            + str(num_classes) + "_bs" + str(batch_size))
    elif model == "dcnn":
        config["model"] = dcnn.DCNN(channel=channel, 
                                    n_class=num_classes)
        config["model_name"] = "DCNN"
        config["model_config"] = (
            "DCNN_is" + str(image_size) + "_ch" + str(channel) + "_nc" \
            + str(num_classes) + "_bs" + str(batch_size))
    elif model == "cnn_lstm":
        config["model"] = xgbfs_cnn_lstm.CNN_LSTM(
            channel=channel, n_classes=num_classes, 
            features=image_size*image_size)
        config["model_name"] = "CNN_LSTM"
        config["model_config"] = (
            "CNN_LSTM_is" + str(image_size) + "_ch" + str(channel) + "_nc" \
            + str(num_classes) + "_bs" + str(batch_size))
    
    # Load saved model if specified
    if continue_train:
        config["model"].load_state_dict(
            torch.load(saved_model_path, 
                        map_location=config["device"]))
        continue_epoch += int(
            saved_model_path.split("epo")[1].split('_')[0])
    config["model_config"] += "_epo" + str(epoch+continue_epoch)

    print(f"Model specified: {config['model_name']}")
    print(f"HYPERPARAMETERS spcified: {HYPERPARAMETERS}")
    return config

# if __name__ == "__main__":
#     set_model.main(standalone_mode=False)
#     model = VitTemplate(hyperparameters=HYPERPARAMETERS)
#     print(f"model: {model}")