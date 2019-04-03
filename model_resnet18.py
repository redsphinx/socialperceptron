from torchvision.models import resnet18
from torch import nn
import torch
from torchvision.models import resnet18


class ResNet18_LastLayers(nn.Module):
    def __init__(self, num_traits):
        super(ResNet18_LastLayers, self).__init__()

        self.fc = nn.Linear(in_features=2*512, out_features=num_traits, bias=True)


    def forward(self, x1, x2):
        h = torch.cat((x1, x2), dim=1)
        h = self.fc(h)
        return h


class hackyResNet18(nn.Module):
    def __init__(self, num_traits, pretrain):
        super(hackyResNet18, self).__init__()
        self.real_resnet18 = resnet18(pretrained=pretrain)
        self.real_resnet18.fc = nn.Sequential()
        self.fc = nn.Linear(in_features=512, out_features=num_traits, bias=True)

    def forward(self, x):
        h = self.real_resnet18(x)
        h = self.fc(h)
        h = torch.tanh(h)
        h = (1 + h) / 2
        return h
