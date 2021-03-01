import torch
import copy
import numpy as np
import random


class EdgeServerNode:
	def __init__(self, model):
		self.model = None
		self.connected_clients = []
		self.location = np.array((random.random(), random.random()))


	def add_client(self, client_id):
		self.connected_clients.append(client_id)


	def average_client_models(self):
		averaged_model = copy.deepcopy(self.model)

		models = [client.model["model"] for client in self.connected_clients]

		with torch.no_grad():
			averaged_values = {}
			for name, param in averaged_model.named_parameters():
				averaged_values[name] = nn.Parameter(torch.zeros_like(param.data))

			for model in models:
				for name, param in model.named_parameters():	
					averaged_values[name] += param.data

			for name, param in averaged_model.named_parameters():
				param.data = (averaged_values[name]/len(models))

		self.model = averaged_model
		