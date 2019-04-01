from torchvision.models import resnet18
from torch import nn


class ResNet18(resnet18(pretrained=True)):
    def __init__(self, num_traits):
        super(ResNet18, self).__init__()
        self.fc = nn.Linear(in_features=512, out_features=num_traits, bias=True)


    def forward(self, x):
        h = self.fc(x)
        h = torch.tanh(h)
        h = (h + 1) / 2
        return h
