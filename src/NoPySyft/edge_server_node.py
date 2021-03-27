import torch
import numpy as np
import random

random.seed(1)

class EdgeServerNode:
	def __init__(self):
		self.model = None
		self.connected_clients = []
		self.neighbor_servers = []
		self.location = np.array((random.random(), random.random()))


	def add_client(self, client_id):
		self.connected_clients.append(client_id)


	def clear_model(self):
		del self.model
		self.model = None


	def sum_model(self):
		total = 0
		for name, param in self.model.named_parameters():
			total += torch.sum(param.data)
		return total