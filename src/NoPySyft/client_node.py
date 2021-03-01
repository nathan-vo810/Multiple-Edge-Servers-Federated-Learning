import random
import numpy as np
import copy
import torch

from mnist_model import CNNModel


class ClientNode:
	def __init__(self, id):
		self.id = id
		self.model = {"model": None, "optim": None, "criterion": None, "loss": None}
		self.data = []
		self.location = self.generate_location()


	def generate_location(self):
		location_ranges = [0,1,2,3,4]
		distributions = [0.1, 0.15, 0.2, 0.25, 0.3]

		index = np.random.choice(location_ranges, 1, replace=False, p=distributions)[0]
		start = 0.2 * index
		end = start + 0.2

		location = (random.uniform(start, end), random.uniform(start, end))

		return location


	def average_models(self):
		averaged_model = copy.deepcopy(self.model["model"][0])

		models = self.model["model"]

		with torch.no_grad():
			averaged_values = {}
			for name, param in averaged_model.named_parameters():
				averaged_values[name] = nn.Parameter(torch.zeros_like(param.data))

			for model in models:
				for name, param in model.named_parameters():	
					averaged_values[name] += param.data

			for name, param in averaged_model.named_parameters():
				param.data = (averaged_values[name]/len(models))

		self.model["model"] = averaged_model


	def clear_model(self):
		del self.model["model"]
		self.model["model"] = None


	def train(self, device):
		if isinstance(self.model["model"], list):
			if len(self.model["model"]) > 1:
				self.average_models()
			else:
				self.model["model"] = self.model["model"][0]

		for batch_idx, (images, labels) in enumerate(self.data):
			images, labels = images.to(device), labels.to(device)

			if (batch_idx+1)%100==0:
				print(f"Processed {batch_idx+1}/{len(self.data)} batches")

			self.model["optim"].zero_grad()
			output = self.model["model"].forward(images)
			loss = self.model["criterion"](output, labels)
			loss.backward()
			self.model["optim"].step()

	
