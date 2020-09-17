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
	def __init__(self, model='resnet18', pretrained=True):
		super(Encoder, self).__init__()
		
		net = models.__dict__[model](pretrained=pretrained)

		self.features = nn.Sequential(*list(net.children())[:-1])
		self.num_ftrs = net.fc.in_features


	def forward(self, x):
		z = self.features(x) #512
		# print(z.shape)
		z = z.squeeze()
		return z

class Autoregressor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoregressor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.autoregressor = nn.GRU(self.input_dim, self.hidden_dim, batch_first=True)



    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.get_device())
        self.autoregressor.flatten_parameters()
        out, _ = self.autoregressor(x, h0)
        return out


class CPC(torch.nn.Module):
	def __init__(
        self, args, model='restnet18', pretrained=True, genc_hidden=512, gar_hidden=128):

		super(CPC, self).__init__()
		self.args = args

		"""
		First, a non-linear encoder genc maps the input sequence of observations xt to a
		sequence of latent representations zt = genc(xt), potentially with a lower temporal resolution.
		"""
		self.encoder = Encoder(model, pretrained)

		"""
		We then use a GRU RNN [17] for the autoregressive part of the model, gar with 256 dimensional hidden state.
		"""
		self.autoregressor = Autoregressor(input_dim=genc_hidden, hidden_dim=gar_hidden)

		self.loss = InfoNCE.InfoNCE(args, gar_hidden, genc_hidden)

	def get_latent_size(self, input_size):
		x = torch.zeros(input_size).to(self.args.device)

		z, c = self.get_latent_representations(x)
		return c.size(2), c.size(1)

	def get_latent_representations(self, x):
		"""
		Calculate latent representation of the input with the encoder and autoregressor
		:param x: inputs (B x L x C X H x W)
		:return: loss - calculated loss
		        accuracy - calculated accuracy
		        z - latent representation from the encoder (B x L x 512)
		        c - latent representation of the autoregressor  (B x 128)
		"""

		# calculate latent represention from the encoder
		z = self.encoder(x) # B x L x 512
		
		# z = z.permute(0, 2, 1)  # swap L and C
		z = self.reshape_input_rnn(z)
		# calculate latent representation from the autoregressor
		c = self.autoregressor(z)
		# print('z shape: ', z.shape) # B x D x 512
		# print('c shape: ', c.shape) # B x 128

		return z, c

	def reshape_input_rnn(self, x):
		x = x.view(-1, 32, 512);
		return x

	def forward(self, x):
		z, c = self.get_latent_representations(x)
		loss, accuracy = self.loss.get(x, z, c)
		return loss, accuracy, z, c