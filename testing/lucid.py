import torch
import torch.nn as nn

class Lucid(nn.Module):
    def __init__(self, *args, n_classes: int = 2, channels: int = 1, 
                 features: int = 16, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_classes = n_classes
        self.channels = channels
        self.features = features
        self.num_kernels = 64

        self.conv = nn.Conv1d(self.features, self.num_kernels, 3, 
                              padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.AdaptiveMaxPool1d(1)

        self.out = nn.Linear(self.num_kernels, self.n_classes)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        batches, self.channels, _, _ = x.size()
        # x = x.view(batches, self.channels, -1)
        x = x.view(batches, self.channels, -1).permute(0, 2, 1)
        # print(f"size of x 0: {x.size()}")

        x = self.conv(x)
        # print(f"size of x 1: {x.size()}")
        x = self.relu(x)
        x = self.maxpool(x)
        # print(f"size of x 2: {x.size()}")

        x = torch.flatten(x, start_dim=1)
        x = self.out(x)
        # print(f"size of x 3: {x.size()}")
        return x