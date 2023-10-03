import torch
import torch.nn as nn

class ANOVA_DNN(nn.Module):
    def __init__(self, *args, channel: int = 1, features: int = 16, 
                 n_classes: int = 2, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.channel = channel
        self.features = features
        self.n_classes = n_classes

        self.fc1 = nn.Linear(self.features, 76)
        self.fc2 = nn.Linear(76, 76)

        self.out = nn.Linear(76, n_classes)
        

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        batches, _, _, _ = x.size()
        x = x.view(batches, -1)

        x = self.fc1(x)
        x = self.fc2(x)

        x = self.out(x)

        return x