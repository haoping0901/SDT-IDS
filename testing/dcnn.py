import torch
import torch.nn as nn

# Reference
# https://github.com/moskomule/senet.pytorch/blob/master/senet/
# se_module.py
class SELayer(nn.Module):
    def __init__(self, channel, reduction):
        super().__init__()
        hidden_neurons = (channel // 2 if channel < reduction 
                          else channel // reduction)

        self.globalpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, hidden_neurons, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_neurons, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.globalpool(x).view(b, -1)
        y = self.fc(y).view(b, c, -1)
        return x * y.expand_as(x)


class DCNN(nn.Module):
    def __init__(self, channels: int = 1, n_classes: int = 2, 
                 features: int = 16):
        super().__init__()
        self.channels = channels
        self.n_classes = n_classes
        self.features = features

        self.num_kernels = 32
        self.kernel_size = 3
        self.kernel_growth_ratio = 2
        self.num_sconv = 2
        self.reduction_ratio = 2

        self.num_grown_kernels = (self.num_kernels 
                                  * self.kernel_growth_ratio)

        # Start building model
        self.conv1 = nn.Conv1d(self.channels, self.num_kernels, 
                               self.kernel_size, padding="same")
        # self.conv1 = nn.Conv1d(self.features, self.num_kernels, 
        #                        self.kernel_size, padding="same")
        self.bn1 = nn.BatchNorm1d(self.num_kernels)
        self.selu1 = nn.SELU(inplace=True)
        self.se1 = SELayer(self.num_kernels, 
                           self.reduction_ratio)
        
        self.conv2 = nn.Conv1d(self.num_kernels, 
                               self.num_kernels, 
                               self.kernel_size,
                               padding="same")
        self.bn2 = nn.BatchNorm1d(self.num_kernels)
        self.selu2 = nn.SELU(inplace=True)
        self.se2 = SELayer(self.num_kernels, 
                           self.reduction_ratio)
        
        self.sconv1 = self._sconv(self.num_kernels, 
                                  self.num_grown_kernels)
        self.se3 = SELayer(self.num_grown_kernels, 
                           self.reduction_ratio)

        sconvi = list()
        for i in range(1, self.num_sconv):
            sconvi.append(self._sconv(self.num_grown_kernels, 
                          self.num_grown_kernels))
            sconvi.append(
                SELayer(self.num_grown_kernels, 
                        self.reduction_ratio)
            )
        self.sconv = nn.Sequential(*sconvi)

        self.globalpool = nn.AdaptiveAvgPool1d(1)
        self.out = nn.Linear(self.num_grown_kernels, self.n_classes)

    def _sconv(self, in_channel: int, out_channel: int
               ) -> nn.Sequential:
        return nn.Sequential(
            # Depth-wise conv
            nn.Conv1d(in_channel, in_channel, 
                      self.kernel_size, padding="same", 
                      groups=in_channel, bias=False),
            nn.BatchNorm1d(in_channel),

            # Point-wise conv
            nn.Conv1d(in_channel, out_channel, 1),
            nn.BatchNorm1d(out_channel),
            nn.SELU(inplace=True),
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # Due to the input shape would be the tabular task format
        # we may need to refit the shape of input
        batch, channels, _, _ = x.size()
        x = x.view(batch, channels, -1)
        # x = x.view(batch, channels, -1).permute(0, 2, 1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.selu1(x)
        x = self.se1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.selu2(x)
        x = self.se2(x)
        x = self.sconv1(x)
        x = self.se3(x)
        x = self.sconv(x)
        x = self.globalpool(x)

        x = torch.flatten(x, start_dim=1)
        x = self.out(x)
        return x