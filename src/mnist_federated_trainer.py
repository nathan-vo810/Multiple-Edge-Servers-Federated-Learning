import torch
import syft

from torch import nn, optim
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset

hook = syft.TorchHook(torch)

from syft.federated.floptimizer import Optims

from mnist_model import Model


class FederatedTrainer:
	def __init__(self, batch_size, lr, num_epochs, model_weight_path, num_workers, iid):
		self.model = Model()
		
		self.lr = lr
		self.batch_size = batch_size
		self.num_epochs = num_epochs
		self.model_weight_path = model_weight_path

		self.workers = self.init_worker(num_workers)

		self.iid = iid

	
	def init_worker(self, num_workers):
		return [syft.VirtualWorker(hook, id=f"worker_{i}") for i in range(num_workers)] 
	
	def load_data(self, train):
		transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
		data = datasets.MNIST('./data', train=train, download=True, transform=transform)
		
		return data


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

	def prepare_federated_iid_data(self, train=True):

		'''
		Distribute the data in batches to workers
		Each worker holds several batches
		'''

		data = self.load_data(train=train)
		
		print("Distributing data...")
		federated_iid_data_loader = syft.FederatedDataLoader(data.federate(self.workers), batch_size=self.batch_size, shuffle=True)
		print("Done!")

		return federated_iid_data_loader

	def prepare_federated_non_iid_data(self, train=True):

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

		data = self.load_data(train=train)

		sorted_datasets = ConcatDataset([get_data_of_number(data, i) for i in range(10)])
		
		images = []
		labels = []

		federated_non_iid_data = []

		print("Distributing data...")
		
		for i in range(len(sorted_datasets)): 		    
		    images.append(normalize(sorted_datasets[i][0]).unsqueeze(0))
		    labels.append(sorted_datasets[i][1])
		    
		    if (i == len(sorted_datasets) - 1) or ((i+1) % 32 == 0) or (sorted_datasets[i][1] != sorted_datasets[i+1][1]):
		        images = torch.stack(images)
		        labels = torch.stack(labels)
		        
		        receiving_worker = self.workers[sorted_datasets[i][1] % len(self.workers)]
		        federated_non_iid_data.append((images.send(receiving_worker), labels.send(receiving_worker)))

		        images = []
		        labels = []

		print("Done!")

		return federated_non_iid_data

	def prepare_data(self, train=False):
		data = self.load_data(train=train)
		data_loader = DataLoader(data, batch_size=self.batch_size, shuffle=True)

		return data_loader

	def train(self):

		if self.iid == True:
			print("Train in Federated IID Mode")
			train_data = self.prepare_federated_iid_data(train=True)
		else:
			print("Train in Federated Non-IID Mode")
			train_data = self.prepare_federated_non_iid_data(train=True)
		
		workers = list()
		for worker in self.workers:
			workers.append(worker.id)
		
		optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
		# optimizer = optimizer.fix_precision()
		optims = Optims(workers, optim=optimizer)
		criterion = nn.CrossEntropyLoss()

		print("Start training...")

		# self.model.fix_precision().share(*self.workers)

		for epoch in range(self.num_epochs):
			self.model.train()

			print("Epoch {}/{}".format(epoch+1, self.num_epochs))

			running_loss = 0
			for batch_idx, (images, labels) in enumerate(train_data):
				self.model = self.model.send(images.location)
				opt = optims.get_optim(images.location.id)

				if batch_idx % 100 == 0: 
					print("Processed {}/{} batches".format(batch_idx, len(train_data)))

				optimizer.zero_grad()

				output = self.model(images)
				
				loss = criterion(output, labels)
				# batch_size = output.shape[0]
				# loss = ((output - target)**2).sum().refresh()/self.batch_size

				loss.backward()
				opt.step()

				# running_loss += loss.get().item()
				self.model.get()

		print("Saving model...")
		torch.save(self.model.state_dict(), MODEL_WEIGHT_PATH)
		print("Finish training!")


	def validate(self):
		print("Start validating...")
		self.model.load_state_dict(torch.load(self.model_weight_path))
		self.model.eval()
		corrects = 0

		test_data = self.prepare_data(train=False)

		with torch.no_grad():
			for batch_idx, (images, labels) in enumerate(test_data):
				print("Predicting batch {}/{}".format(batch_idx+1, len(test_data)))
				output = self.model(images)
				pred = output.argmax(dim=1)
				corrects += pred.eq(labels.view_as(pred)).sum().item()

		print("Number of corrects: {}/{}".format(corrects, len(test_data)*self.batch_size))
		print("Accuracy: {}%".format(100*corrects/(len(test_data)*self.batch_size)))