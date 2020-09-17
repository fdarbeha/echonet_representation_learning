import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import numpy as np
from torch.cuda.amp import autocast
import torchvision.models as models
import losses.infoNCE as InfoNCE
from torch.cuda.amp import autocast


class Encoder(nn.Module):
	def __init__(self, model='r3d_18', pretrained=True):
		super(Encoder, self).__init__()

		net = torchvision.models.video.__dict__[model]\
						(pretrained=pretrained)

		self.features = nn.Sequential(*list(net.children())[:-1])
		self.num_ftrs = net.fc.in_features


	def forward(self, x):
		z = self.features(x) #512
		z = z.squeeze()
		return z, 0



def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)

class ResizeConv3d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
		super().__init__()
		self.scale_factor = scale_factor
		self.mode = mode
		self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=1,padding=1)

	def forward(self, x):
		x = F.interpolate(x, scale_factor=self.scale_factor,
							mode = self.mode)
		x = self.conv(x)

		return x

class BasicDecoderBlock(nn.Module):
	def __init__(self, in_planes, stride=1, out_channels=None):
		super(BasicDecoderBlock, self).__init__()

		if out_channels != None and stride!= 1:
			planes = out_channels
		else:
			planes = int(in_planes/stride)
		
		self.conv1 = ResizeConv3d(in_planes, planes, kernel_size=3, scale_factor=stride)
		self.bn1 = nn.BatchNorm3d(planes)
		self.shortcut = nn.Sequential(ResizeConv3d(in_planes, planes, kernel_size=3, scale_factor=stride),
										nn.BatchNorm3d(planes))

		self.conv2 = conv3x3x3(in_planes, in_planes, stride=1)
		self.bn2 = nn.BatchNorm3d(in_planes)



	def forward(self, x):
		out = torch.relu(self.bn2(self.conv2(x)))
		out = self.bn1(self.conv1(out))
		out += self.shortcut(x)
		out = torch.relu(out)

		return out

class Decoder(nn.Module):
	def __init__(self, num_blocks=[2, 2, 2, 2], z_dim=10, nc=3):
		super().__init__()
		self.in_planes = 512

		self.linear = nn.Linear(z_dim, 512)

		self.layer4 = self._make_layer(BasicDecoderBlock, 256, num_blocks[3], stride=2)
		self.layer3 = self._make_layer(BasicDecoderBlock, 128, num_blocks[2], stride=2)
		self.layer2 = self._make_layer(BasicDecoderBlock, 64, num_blocks[1], stride=2)
		self.layer1 = self._make_layer(BasicDecoderBlock, 32, num_blocks[0], stride=2)
		self.conv1 = ResizeConv3d(32, nc, kernel_size=3, scale_factor=1)
		self.conv2 = conv3x3x3(112, 112, stride=1)



	def _make_layer(self, BasicDecoderBlock, planes, num_blocks, stride, out_channels=None):
		strides = [stride] + [1]*(num_blocks - 1)
		layers = []
		for stride in reversed(strides):
			layers += [BasicDecoderBlock(self.in_planes, stride, out_channels)]

		self.in_planes = planes
		return nn.Sequential(*layers)

	def forward(self, x):
		# print('input: ', x.shape)
		x = x.view(x.size(0), 512, 1, 1, 1)
		# print('view: ', x.shape)
		x = F.interpolate(x, scale_factor=4)
		# print('interpolate: ', x.shape)
		x = self.layer4(x)
		# print('layer 4: ', x.shape)
		x = self.layer3(x)
		# print('layer 3: ', x.shape)
		x = self.layer2(x)
		# print('layer 2: ', x.shape)
		x = self.layer1(x)
		# print('layer 1: ', x.shape)

		x = torch.sigmoid(self.conv1(x))
		# print('conv1: ', x.shape)
		x = F.interpolate(x, size=[32, 112, 112])
		# print('interpolate2 : ', x.shape)
		x = x.view(x.size(0), 3, 32, 112, 112)

		return x





