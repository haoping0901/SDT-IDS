import torch
import torch.nn as nn

class AEMLP(nn.Module):
    def __init__(self, n_class: int = 2):
        super().__init__()
        self.n_class = n_class
        # Autoencoder part
        self.ae1 = nn.Linear(77, 32)
        self.relu1 = nn.ReLU(inplace=True)
        self.ae2 = nn.Linear(32, 24)
        self.relu2 = nn.ReLU(inplace=True)

        # MLP part
        self.mlp1 = nn.Linear(24, 23)
        self.relu3 = nn.ReLU(inplace=True)

        self.mlp2 = nn.Linear(23, 15)
        self.relu4 = nn.ReLU(inplace=True)

        self.mlp3 = nn.Linear(15, 10)
        self.relu5 = nn.ReLU(inplace=True)

        self.output = nn.Linear(10, self.n_class)


    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # batch, _, _, _ = x.size()
        # x = x.view(batch, -1)
        
        x = self.ae1(x)
        x = self.relu1(x)
        x = self.ae2(x)
        x = self.relu2(x)
        x = self.mlp1(x)
        x = self.relu3(x)
        x = self.mlp2(x)
        x = self.relu4(x)
        x = self.mlp3(x)
        x = self.relu5(x)
        x = self.output(x)
        
        return x