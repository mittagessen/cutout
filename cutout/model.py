import torch

from torchvision import models
import torch.nn.functional as F
from torch import autograd

from torch import nn


def _wi(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.LSTM):
        for p in m.parameters():
            # weights
            if p.data.dim() == 2:
                torch.nn.init.orthogonal_(p.data)
            # initialize biases to 1 (jozefowicz 2015)
            else:
                torch.nn.init.constant_(p.data[len(p)//4:len(p)//2], 1.0)
    elif isinstance(m, torch.nn.GRU):
        for p in m.parameters():
            torch.nn.init.orthogonal_(p.data)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        torch.nn.init.constant_(m.bias, 0)

class ClassificationNet(nn.Module):
    """
    ResNet-152
    """
    def __init__(self, refine_features=False):
        super(ClassificationNet, self).__init__()
        self.resnet = models.resnet152(pretrained=True)
        self.refine_features(refine_features)
        self.fc = nn.Linear(self.resnet.fc.in_features, 1)
        self.adamaxpool = nn.AdaptiveMaxPool2d(1)

    def refine_features(self, refine_features):
        if not refine_features:
            for param in self.resnet.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = x.permute(0, 2, 3, 1)
        x = self.fc(x).permute(0, 3, 1, 2)
        x = self.adamaxpool(x)
        return x.squeeze().unsqueeze(0)

    def init_weights(self):
        self.fc.apply(_wi)
