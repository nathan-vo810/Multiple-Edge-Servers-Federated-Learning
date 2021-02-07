import torch
import torch.nn.functional as F

from torch import nn


class Model(nn.Module):
	'''
	CNN model with two 5x5 convolution layers. 
	The first layer has 20 output channels and the second has 50, with each layer followed by 2x2 max pooling.
	On each device, the batch size is ten and the epoch number is five.
	'''
	def __init__(self):
		super().__init__()

		self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=2)
		self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=2)
		self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
		self.flatten = nn.Flatten()
		self.fc = nn.Linear(2450, 10)

	def forward(self, x):
		x = self.maxpool(F.relu(self.conv1(x)))
		x = self.maxpool(F.relu(self.conv2(x)))
		x = self.fc(self.flatten(x))

		return x