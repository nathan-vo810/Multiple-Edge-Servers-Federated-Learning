import torch
import syft

from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset

class MNIST_DataLoader:
	def __init__(self, batch_size, workers):
		self.train_data = self.load_data(train=True)
		self.test_data = self.load_data(train=False)

		self.batch_size = batch_size
		self.workers = workers


	def load_data(self, train):
		transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
		data = datasets.MNIST('../data', train=train, download=True, transform=transform)
		
		return data


	def prepare_data(self, train):
		if train == True:
			data_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
		else:
			data_loader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=True)

		return data_loader


	def prepare_federated_iid_data_sequential(self, train=True):

		'''
		Distribute the data in batches to workers
		Each worker holds several batches
		'''

		if train == True:
			data = self.train_data
		else:
			data = self.test_data
		
		print("Distributing data...")
		federated_iid_data_loader = syft.FederatedDataLoader(data.federate(self.workers), batch_size=self.batch_size, shuffle=True)
		print("Done!")

		return federated_iid_data_loader


	def prepare_federated_iid_data_parallel(self, train=True):

		'''
		Distribute the data in batches to workers
		Each worker holds several batches
		'''

		if train == True:
			data = self.train_data
		else:
			data = self.test_data
		
		print("Distributing data...")
		federated_iid_data_loader = syft.FederatedDataLoader(data.federate(self.workers), batch_size=self.batch_size, shuffle=True)
		print("Done!")

		return federated_iid_data_loader


	def prepare_federated_non_iid_data_sequential(self, train=True):

		'''
		Sort data
		Distribute the classes to workers
		Each worker holds the complete data of a class
		'''

		def get_data_of_number(data, number):
			indices = (data.targets == number).int()

			images = data.data[indices == 1]
			labels = data.targets[indices == 1]

			return TensorDataset(images, labels)

		def normalize(x, mean=0.1307, std=0.3081):
			return (x-mean)/std

		if train == True:
			data = self.train_data
		else:
			data = self.test_data

		data_by_class = [get_data_of_number(data, i) for i in range(10)]

		workers_classes = {}
		for number in range(10):
			class_data = get_data_of_number(data, number)

			receiving_worker = number%len(self.workers)
			if receiving_worker not in workers_classes:
				workers_classes[receiving_worker] = [class_data]
			else:
				workers_classes[receiving_worker].append(class_data)

		for worker_number, worker_data in workers_classes.items():
			workers_classes[worker_number] = DataLoader(ConcatDataset(worker_data), shuffle=True, batch_size=self.batch_size)


		federated_non_iid_data = {}
		for worker_number in range(len(self.workers)):
			federated_non_iid_data[worker_number] = []

		print("Distributing data...")
		for worker_number, worker_data in workers_classes.items():
			print(f"Sending data to worker_{worker_number}")
			worker = self.workers[worker_number]
			for batch_idx, (images, labels) in enumerate(worker_data):
				images = normalize(images).unsqueeze(1)
				federated_non_iid_data[worker_number].append((images.send(worker), labels.send(worker)))
		print("Done!")

		return federated_non_iid_data


	def prepare_federated_non_iid_data_parallel(self, train=True):

		'''
		Sort data
		Distribute the classes to workers
		Each worker holds the complete data of a class
		'''

		def get_data_of_number(data, number):
			indices = (data.targets == number).int()

			images = data.data[indices == 1]
			labels = data.targets[indices == 1]

			return TensorDataset(images, labels)

		def normalize(x, mean=0.1307, std=0.3081):
			return (x-mean)/std

		if train == True:
			data = self.train_data
		else:
			data = self.test_data

		data_by_class = [get_data_of_number(data, i) for i in range(10)]

		workers_classes = {}
		for number in range(10):
			class_data = get_data_of_number(data, number)

			receiving_worker = number%len(self.workers)
			if receiving_worker not in workers_classes:
				workers_classes[receiving_worker] = [class_data]
			else:
				workers_classes[receiving_worker].append(class_data)

		for worker_number, worker_data in workers_classes.items():
			workers_classes[worker_number] = DataLoader(ConcatDataset(worker_data), shuffle=True, batch_size=self.batch_size)


		federated_non_iid_data = {}
		for worker_number in range(len(self.workers)):
			federated_non_iid_data[worker_number] = []

		print("Distributing data...")
		for worker_number, worker_data in workers_classes.items():
			print(f"Sending data to worker_{worker_number}")
			worker = self.workers[worker_number]
			for batch_idx, (images, labels) in enumerate(worker_data):
				images = normalize(images).unsqueeze(1)
				federated_non_iid_data[worker_number].append((images.send(worker), labels.send(worker)))
		print("Done!")

		return federated_non_iid_data



	def prepare_private_data(self, precision_fractional=3, train=True):

		'''
		Each batch is shared across workers
		Each worker hold parts of batches
		'''

		def one_hot_of(index_tensor):
			onehot_tensor = torch.zeros(*index_tensor.shape, 10)
			onehot_tensor = onehot_tensor.scatter(1, index_tensor.view(-1, 1), 1)
			return onehot_tensor

		def secret_share(tensor):
			return (tensor.fix_precision(precision_fractional=precision_fractional).share(*self.workers, protocol="fss", requires_grad=True))

		data = self.load_data(train=train)
		data_loader = DataLoader(data, batch_size=self.batch_size)

		print("Distributing data...")
		private_data_loader = [(secret_share(images), secret_share(one_hot_of(labels))) for images, labels in data_loader]	
		print("Done!")

		return private_data_loader		