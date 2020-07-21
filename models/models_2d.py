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
	def __init__(self, model='resnet18', pretrained=False, z_dimension=128):
		super(Encoder, self).__init__()
		
		net = models.__dict__[model](pretrained=pretrained)

		self.features = nn.Sequential(*list(net.children())[:-1])
		self.num_ftrs = net.fc.in_features
		# Projection head
		self.l1 = nn.Linear(self.num_ftrs, self.num_ftrs)
		self.l2 = nn.Linear(self.num_ftrs, z_dimension)


	def forward(self, x):
		z = self.features(x) #512
		# print(z.shape)
		z = z.squeeze()

		h_p = self.l1(z)
		h_p = F.relu(h_p)
		h_p = self.l2(h_p)

		return z, h_p

class Autoregressor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoregressor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.autoregressor = nn.GRU(self.input_dim, self.hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
        self.activation = nn.Sigmoid()



    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.get_device())
        self.autoregressor.flatten_parameters()
        out, _ = self.autoregressor(x, h0)

        out = self.linear(out[:, -1, :])
        out = self.activation(out)
        return out

class Classifier_2D(nn.Module):
	def __init__(self, model='resnet18', pretrained=True, genc_hidden=512, gar_hidden=128):
		super(Classifier_2D, self).__init__()

		self.encoder = Encoder(model, pretrained)
		self.rnn = Autoregressor(genc_hidden, gar_hidden)

	def reshape_input_rnn(self, x):
		x = x.view(-1, 32, 512);
		# print(x.shape)
		return x

	def forward(self, x):
		out, _ = self.encoder(x)
		out = self.reshape_input_rnn(out)
		out = self.rnn(out)

		return out

