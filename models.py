import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class GazeNet(nn.Module):

    def __init__(self):
        super(GazeNet, self).__init__()
        res = models.resnet18(pretrained=True)

        self.features = nn.Sequential(
            res.conv1,
            res.bn1,
            res.relu,
            res.maxpool,
            res.layer1,
            res.layer2,
            res.layer3,
            res.layer4
        )

        # self.Conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)
        # self.Conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)
        # self.Conv3 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.layer1 = nn.Sequential(
            nn.Linear(512*8*8, 4096),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )   
        self.fc = nn.Linear(512, 2)

    #     self._initialize_weight()
    #     self._initialize_bias()

    # def _initialize_weight(self):
    #     nn.init.normal_(self.Conv1.weight, mean=0.0, std=0.01)
    #     nn.init.normal_(self.Conv2.weight, mean=0.0, std=0.01)
    #     nn.init.normal_(self.Conv3.weight, mean=0.0, std=0.001)

    # def _initialize_bias(self):
    #     nn.init.constant_(self.Conv1.bias, val=0.1)
    #     nn.init.constant_(self.Conv2.bias, val=0.1)
    #     nn.init.constant_(self.Conv3.bias, val=1)

    def forward(self, x):
        x = self.features(x)
        
        # y = F.relu(self.Conv1(x))
        # y = F.relu(self.Conv2(y))
        # y = F.relu(self.Conv3(y))

        # x = F.dropout(F.relu(torch.mul(x, y)), 0.5)
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.fc(x)

        return x