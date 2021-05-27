# -*- coding:utf-8 -*-

from torch import nn
from torchvision import models
import torch


class Net(nn.Module):
    def __init__(self):
        """Declare all needed layers."""
        super(Net, self).__init__()
        resnet = models.resnet50()
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.Dropout(0.4),
            nn.LeakyReLU(),
            nn.Linear(1024, 256),
            nn.Dropout(0.4),
            nn.LeakyReLU(),
        )

        self.fc_s = nn.Sequential(nn.Linear(256, 1))

    def forward(self, X):
        X1, X2 = X.chunk(2, 1)
        x1 = X1.contiguous().view(
            X1.size()[0] * X1.size()[1], X1.size()[2], X1.size()[3], X1.size()[4]
        )
        x2 = X2.contiguous().view(
            X2.size()[0] * X2.size()[1], X2.size()[2], X2.size()[3], X2.size()[4]
        )

        x1 = self.features(x1)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.fc(x1)
        x1_s = self.fc_s(x1)

        x2 = self.features(x2)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.fc(x2)
        x2_s = self.fc_s(x2)

        x1_s = x1_s.view(-1, 3).mean(dim=1)
        x2_s = x2_s.view(-1, 3).mean(dim=1)
        s = torch.stack((x1_s, x2_s), dim=1).mean(dim=1)
        return s
