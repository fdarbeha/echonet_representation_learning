import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import numpy as np
from torch.cuda.amp import autocast



class ENC3D(nn.Module):
	def __init__(self, model='r3d_18', pretrained=True, z_dimension=128, mode='projection_head'):
		super(ENC3D, self).__init__()
		self.mode = mode

		enc_3d = torchvision.models.video.__dict__[model]\
						(pretrained=pretrained)

		self.features = nn.Sequential(*list(enc_3d.children())[:-1])
		self.num_ftrs = enc_3d.fc.in_features

		# projection MLP
		self.l1 = nn.Linear(self.num_ftrs, self.num_ftrs)
		self.l2 = nn.Linear(self.num_ftrs, z_dimension)


	@autocast()
	def forward(self, x):
		h = self.features(x) #512
		h = h.squeeze()

		
		h_p = self.l1(h)
		h_p = F.relu(h_p)
		h_p = self.l2(h_p)


		return h, h_p



def construct_3d_enc(model='r3d_18', pretrained=True, z_dimension=128, mode='projection_head'):
	return ENC3D(model, pretrained, z_dimension, mode)


class LinearRegressor(nn.Module):
	def __init__(self, model, num_ftrs, z_dimension=128):
		super(LinearRegressor, self).__init__()
		self.num_ftrs = num_ftrs
		self.regressor1 = nn.Linear(self.num_ftrs, 1)
		self.regressor2 = nn.Linear(z_dimension, 1)
		self.activation = nn.Sigmoid()

	@autocast()
	def forward(self, x):
		out = x
		out = self.activation(self.regressor1(out))

		return out

def construct_linear_regressor(model, z_dimension=128):
	return LinearRegressor(model, 512, z_dimension)




