from testing import lnn, dcnn, aemlp, xgbfs_cnn_lstm
from torchinfo import summary

MODEL = "CNN_LSTM"
CHANNEL = 1
CLASSES = 2
BATCH_SIZE = 1
IMAGE_SIZE = 4
ORIGIN_FEATURES = 77

if __name__ == "__main__":
    match MODEL:
        case "LNN":
            model = lnn.LNN(channel=CHANNEL, n_class=CLASSES)
            input_size = (BATCH_SIZE, CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        case "DCNN":
            model = dcnn.DCNN(channel=CHANNEL, n_class=CLASSES)
            input_size = (BATCH_SIZE, CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        case "AEMLP":
            model = aemlp.AEMLP(n_class=CLASSES)
            input_size = (BATCH_SIZE, ORIGIN_FEATURES)
        case "CNN_LSTM":
            model = xgbfs_cnn_lstm.CNN_LSTM(channel=CHANNEL, 
                                            n_classes=CLASSES)
            input_size = (BATCH_SIZE, CHANNEL, IMAGE_SIZE, IMAGE_SIZE)

    summary(
        model=model, 
        input_size=input_size, 
        device="cpu", 
        show_unit=1)
    
        