import torch
import torch.nn as nn

class FLOW1(nn.Module):
    def __init__(self, *args, in_channels: int = 8, 
                 out_channels: int = 8, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, 
                               (3, 1), padding="same")
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2_1 = nn.Conv2d(self.out_channels, self.out_channels, 
                                 (1, 3), padding="same")
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(self.out_channels, self.out_channels, 
                                 (3, 1), padding="same")
        self.relu2_2 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d((2, 1), stride=2)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.conv1(x)
        x = self.relu1(x)
        x2_1 = self.conv2_1(x)
        x2_1 = self.relu2_1(x2_1)
        x2_2 = self.conv2_2(x)
        x2_2 = self.relu2_2(x2_2)

        x = torch.cat([x2_1, x2_2], 1)
        x = self.maxpool(x)

        return x

class CNN_LSTM(nn.Module):
    def __init__(self, *args, channel: int = 1, n_classes: int = 2, 
                 features: int = 16, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.channel = channel
        self.n_classes = n_classes
        self.features = features
        self.channel_expanded = 8

        # Start building model
        self.bn1 = nn.BatchNorm2d(self.channel)
        self.conv1 = nn.Conv2d(self.channel, self.channel_expanded, 
                               (1, 3), padding="same")
        self.relu1 = nn.ReLU(inplace=True)

        self.flow1 = FLOW1()
        self.flow2 = self._flow2(self.channel_expanded, 
                                 self.channel_expanded)
        self.flow3 = nn.MaxPool2d((2, 1), stride=2)
        
        # input channel: 16 from FLOW1, 16 from _flow2
        self.conv2 = nn.Conv2d(self.channel_expanded*4, 
                               self.channel_expanded, (1, 1), 
                               padding="same")
        self.relu2 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d((2, 1), 2)
        self.lstm = nn.LSTM(input_size=8, hidden_size=25, 
                            batch_first=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(features//2//2*25, self.n_classes)

    def _flow2(self, in_channels: int = 8, out_channels: int = 8
               ) -> torch.FloatTensor:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 3), 
                      padding="same"), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(out_channels, out_channels*2, (3, 1), 
                      padding="same"),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d((2, 1), stride=2)
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        batches, channels, _, _ = x.size()
        x = x.view(batches, channels, -1, 1)

        x = self.bn1(x)
        x = self.conv1(x)
        x = self.relu1(x)
        
        x1 = self.flow1(x)
        x2 = self.flow2(x)
        x3 = self.flow3(x)

        x = torch.cat([x1, x2], dim=1)
        x = self.conv2(x)
        x += x3
        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        x, _ = self.lstm(torch.permute(x, (0, 2, 1)))
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x