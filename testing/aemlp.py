import torch
import torch.nn as nn

ENCODED_DIM = 24

class AE_Encoder(nn.Module):
    def __init__(self, *args, in_features: int = 77, 
                 channels: int = 1, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.in_features = in_features
        self.channels = channels

        self.enc1 = nn.Linear(self.in_features*self.channels, 32)
        self.relu1 = nn.ReLU(inplace=True)
        self.enc2 = nn.Linear(32, ENCODED_DIM)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        batches, _, _, _ = x.size()
        x = x.view(batches, -1)

        x = self.enc1(x)
        x = self.relu1(x)
        x = self.enc2(x)
        x = self.relu2(x)

        return x
    

class AE_Decoder(nn.Module):
    def __init__(self, *args, in_features: int = 77, 
                 channels: int = 1, input_img: bool = True, **kwargs
                 ) -> None:
        super().__init__(*args, **kwargs)

        self.in_features = in_features
        self.channels = channels
        self.input_img = input_img

        self.dec1 = nn.Linear(24, 32)
        self.relu1 = nn.ReLU(inplace=True)
        self.dec2 = nn.Linear(32, self.in_features*self.channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.dec1(x)
        x = self.relu1(x)
        x = self.dec2(x)
        x = self.relu2(x)

        if self.input_img:
            batches = x.size()[0]
            x = x.view(batches, self.channels, self.in_features**0.5, 
                    self.in_features**0.5)
        
        return x

class MLP(nn.Module):
    def __init__(self, *args, n_classes: int = 2, 
                 in_features: int = 16, input_img: bool = True, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_classes = n_classes
        self.in_features = in_features
        self.input_img = input_img

        self.mlp1 = nn.Linear(self.in_features, 23)
        self.relu1 = nn.ReLU(inplace=True)

        self.mlp2 = nn.Linear(23, 15)
        self.relu2 = nn.ReLU(inplace=True)

        self.mlp3 = nn.Linear(15, 10)
        self.relu3 = nn.ReLU(inplace=True)

        self.out = nn.Linear(10, self.n_classes)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        if self.input_img:
            batches, _, _, _ = x.size()
            x = x.view(batches, -1)
        
        x = self.mlp1(x)
        x = self.relu1(x)
        x = self.mlp2(x)
        x = self.relu2(x)
        x = self.mlp3(x)
        x = self.relu3(x)
        
        x = self.out(x)

        return x

class AEMLP(nn.Module):
    def __init__(self, n_classes: int = 2, in_features: int = 77, 
                 channels: int = 1):
        super().__init__()
        self.n_classes = n_classes
        self.in_features = in_features
        self.channels = channels

        # Autoencoder part
        self.enc = AE_Encoder(in_features=self.in_features, 
                              channels=self.channels)

        # MLP part
        self.mlp = MLP(n_classes=self.n_classes, 
                       in_features=ENCODED_DIM)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.enc(x)
        x = self.mlp(x)
        
        return x