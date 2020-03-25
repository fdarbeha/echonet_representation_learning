import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import numpy as np



class ENC3D(nn.Module):
    def __init__(self, model='r3d_18', pretrained=True, z_dimention=128, mode='projection_head'):
        super(ENC3D, self).__init__()
        self.mode = mode
        self.enc_3d = torchvision.models.video.__dict__[model]\
        				(pretrained=pretrained)
        self.enc_3d.fc = torch.nn.Linear(self.enc_3d.fc.in_features, z_dimention)

        self.projection_head = nn.Linear(z_dimention, z_dimention)

    def forward(self, x):
        out = self.enc_3d(x)
        if self.mode == 'projection_head':
            out = F.relu(self.projection_head(out))
        return out


def construct_3d_enc(model='r3d_18', pretrained=True, z_dimention=128, mode='projection_head'):
	return ENC3D(model, pretrained, z_dimention, mode)
