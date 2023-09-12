import torch
import torch.nn as nn

# Reference
# https://ieeexplore.ieee.org/document/9566308

class UnitB(nn.Module):
    def __init__(self, num_channels: int, kernel_size: int) -> None:
        super().__init__()

        branch_channel = num_channels//2

        self.branch1 = nn.Sequential(
            # Point-wise conv
            nn.Conv1d(branch_channel, num_channels, 1),
            nn.BatchNorm1d(num_channels),
            nn.ReLU(inplace=True),

            # Depth-wise conv
            nn.Conv1d(num_channels, num_channels, kernel_size, 
                      padding="same", groups=num_channels, 
                      bias=False), 
            nn.BatchNorm1d(num_channels), 

            # standard conv
            nn.Conv1d(num_channels, branch_channel, 1), 
            nn.BatchNorm1d(branch_channel),
            nn.ReLU(inplace=True),
        )
          
    def _concat(self, branch1: torch.FloatTensor, 
                branch2: torch.FloatTensor) -> torch.FloatTensor:
        # Concatenate along channel axis
        return torch.cat((branch1, branch2), 1)

    def _channel_shuffle(self, x: torch.FloatTensor, 
                         groups: int) -> torch.FloatTensor:
        batchsize, num_channels, features = x.shape

        channels_per_group = num_channels // groups
        
        # reshape
        x = x.view(batchsize, groups, channels_per_group, features)

        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batchsize, -1, features)

        return x
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x1 = x[:, :(x.shape[1]//2), :]
        x2 = x[:, (x.shape[1]//2):, :]
        out = self._concat(x1, self.branch1(x2))

        return self._channel_shuffle(out, 2)


class LNN(nn.Module):
    def __init__(self, *args, channel: int = 1, n_class: int = 2, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_kernels = 16
        self.kernel_size = 3
        self.kernel_growth_ratio = 4
        self.num_unitb = 2
        self.expansion_ratio = 1

        self.num_grown_kernels = (self.num_kernels 
                                  * self.kernel_growth_ratio)

        # Start building model
        self.conv1 = nn.Conv1d(channel, self.num_kernels, 
                               self.kernel_size, padding="same")
        self.bn1 = nn.BatchNorm1d(self.num_kernels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.unit_a = self._unit_a(
            self.num_kernels, 
            self.num_grown_kernels, 
        )

        unit_bi = list()
        for i in range(self.num_unitb):
            unit_bi.append(
                UnitB(self.num_grown_kernels, 
                      self.kernel_size)
            )
        self.unit_b = nn.Sequential(*unit_bi)

        self.globalpool = nn.AdaptiveAvgPool1d(1)
        self.output = nn.Linear(self.num_grown_kernels, n_class)

    def _unit_a(self, in_channel: int, out_channel: int, 
                ) -> nn.Sequential:
        kernel_expanded = out_channel*self.expansion_ratio
        return nn.Sequential(
            # Point-wise conv
            nn.Conv1d(in_channel, kernel_expanded, 1), 
            nn.BatchNorm1d(kernel_expanded),
            nn.ReLU(inplace=True),

            # Depth-wise conv
            nn.Conv1d(kernel_expanded, kernel_expanded, 
                      self.kernel_size, stride=2, 
                      groups=kernel_expanded, bias=False), 
            nn.BatchNorm1d(kernel_expanded),

            # standard conv
            nn.Conv1d(kernel_expanded, out_channel, 1),
            nn.BatchNorm1d(out_channel),
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        batch, channels, _, _= x.size()
        x = x.view(batch, channels, -1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.unit_a(x)
        x = self.unit_b(x)
        x = self.globalpool(x)

        x = torch.flatten(x, start_dim=1)
        x = self.output(x)
        return x