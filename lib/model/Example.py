import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


'''
Implement your network
'''
class ExampleNet(nn.Module):
    def __init__(self, backbone='resnet18', n_classes=10, dropout=0.):
        super(ExampleNet, self).__init__()
        self.encoder = getattr(models, backbone)(pretrained=True)
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(128, n_classes, 1),
        )

    def forward(self, x):
        hw = x.shape[2:]
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)

        output = self.decoder(x)
        output = F.interpolate(output, size=hw, mode='bilinear', align_corners=False)
        return {'output': output}

    def compute_losses(self, batch):
        '''
        The batch is defined by your implemented dataset under lib/dataset/

        This function should return a dictionary where all entries are log and show after each epoch.
        The return dict with key 'total' should contain the loss for back-propagation.
        '''
        pred = self.forward(batch['x'])

        losses = {}
        losses['accuracy'] = (pred['output'].argmax(1) == batch['gt']).float().mean()
        losses['total'] = F.cross_entropy(pred['output'], batch['gt'])
        return losses

