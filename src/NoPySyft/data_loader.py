from tqdm import tqdm

import torch

from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset


class MNISTDataLoader:
	def __init__(self, batch_size):
		self.train_data = self.load_data(train=True)
		self.test_data = DataLoader(self.load_data(train=False), batch_size=batch_size, shuffle=True)

		self.batch_size = batch_size


	def load_data(self, train):
		transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
		data = datasets.MNIST('../data', train=train, download=True, transform=transform)
		
		return data


	def prepare_federated_pathological_non_iid(self, no_clients, train=True):
		'''
		Sort the data by digit label
		Divide it into 200 shards of size 300
		Assign each of n clients 200/n shards
		'''

		def normalize(x, mean=0.1307, std=0.3081):
			return (x-mean)/std

		data = self.train_data if train == True else self.test_data
		
		sorted_images = []
		sorted_labels = []

		for number in range(10):
			indices = (data.targets == number).int()

			images = data.data[indices == 1]
			labels = data.targets[indices == 1]

			images = normalize(images).unsqueeze(1)

			sorted_images += images.unsqueeze(0)
			sorted_labels += labels.unsqueeze(0)

		# sorted_images = normalize(sorted_images).unsqueeze(1)

		sorted_images = torch.cat(sorted_images)
		sorted_labels = torch.cat(sorted_labels)


		shards = []
		for i in range(200):
			start = i*300
			end = start+300

			images = sorted_images[start:end]
			labels = sorted_labels[start:end]

			shard = TensorDataset(images, labels)
			shards.append(shard)

		clients_data = {}
		shards_per_client = len(shards)/no_clients
		for shard_idx, shard in enumerate(shards):
			receiving_client = (int)(shard_idx//shards_per_client)
			if receiving_client not in clients_data:
				clients_data[receiving_client] = [shard]
			else:
				clients_data[receiving_client].append(shard)

		for client_number, client_data in clients_data.items():
			clients_data[client_number] = DataLoader(ConcatDataset(client_data), shuffle=True, batch_size=self.batch_size)

		return clients_data