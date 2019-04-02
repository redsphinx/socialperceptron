from torchvision.models import resnet18
from torch import nn
import torch


class ResNet18_LastLayers(nn.Module):
    def __init__(self, num_traits):
        super(ResNet18_LastLayers, self).__init__()

        self.fc = nn.Linear(in_features=2*512, out_features=num_traits, bias=True)


    def forward(self, x1, x2):
        h = torch.cat((x1, x2), dim=1)
        h = self.fc(h)
        return h
