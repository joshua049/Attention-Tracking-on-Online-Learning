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
            nn.Linear(512, 128),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(128, 16),
            nn.ReLU()
        )   
        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        x = self.features(x)
        
        x = F.adaptive_avg_pool2d(x, (1,1))

        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.fc(x)

        return x