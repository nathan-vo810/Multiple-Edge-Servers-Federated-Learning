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
	def __init__(self, batch_size, lr, num_rounds, num_epochs, model_weight_path, num_workers, iid):
		self.model = Model()
		
		self.lr = lr
		self.batch_size = batch_size
		self.num_rounds = num_rounds
		self.num_epochs = num_epochs
		self.model_weight_path = model_weight_path

		self.workers = self.init_worker(num_workers)
		self.secure_worker = self.init_secure_worker()

		self.iid = iid

	
	def init_worker(self, num_workers):
		return [syft.VirtualWorker(hook, id=f"worker_{i}") for i in range(num_workers)]

	def init_secure_worker(self):
		return syft.VirtualWorker(hook, id="secure_worker")
	
	def load_data(self, train):
		transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
		data = datasets.MNIST('../data', train=train, download=True, transform=transform)
		
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

	def prepare_data(self, train=True):
		data = self.load_data(train=train)
		data_loader = DataLoader(data, batch_size=self.batch_size, shuffle=True)

		return data_loader


	def train_parallel(self):

		'''
		Train the model in parallel
		Each worker will train on its own data for several epochs
		Then, it sends the trained model to the secure worker for averaging the weight
		The averaged weight is sent back to all the workers
		This process repeats in several rounds

		'''

		def model_averaging(local_models):
			with torch.no_grad():
				averaged_values = {}
				for name, param in self.model.named_parameters():
					averaged_values[name] = nn.Parameter(torch.zeros_like(param.data))

				for local_model in local_models:
					for name, local_param in local_model.named_parameters():
						averaged_values[name] += local_param.get().data

				for name, param in self.model.named_parameters():
					param.data = (averaged_values[name]/len(local_models))

		def create_local_models():
			worker_models = []
			worker_optims = []
			worker_criterions = []
			worker_losses = []

			for i in range(len(self.workers)):
				model_clone = self.model.copy()

				worker_models.append(model_clone)
				worker_optims.append(optim.Adam(model_clone.parameters(), lr=self.lr))
				worker_criterions.append(nn.CrossEntropyLoss())

			return worker_models, worker_optims, worker_criterions, worker_losses

		if self.iid == True:
			print("Train in Federated IID Mode")
			train_data = self.prepare_federated_iid_data(train=True)
		else:
			print("Train in Federated Non-IID Mode")
			train_data = self.prepare_federated_non_iid_data(train=True)

		print("Start training...")

		for round_iter in range(self.num_rounds):
			print(f"Round {round_iter+1}/{self.num_rounds}")
			worker_models, worker_optims, worker_criterions, worker_losses = create_local_models()

			# Train each worker with its own local data
			for i in range(len(worker_models)):
				print(f"Training worker_{i}")

				# Send model to worker
				worker_models[i] = worker_models[i].send(self.workers[i])

				# Train worker's model
				for epoch in range(self.num_epochs):
					print(f"Epoch {epoch+1}/{self.num_epochs}")
					for batch_idx, (images, labels) in enumerate(train_data[i]):
						if (batch_idx+1)%100==0:
							print(f"Processed {batch_idx+1}/{len(train_data[i])} batches")

						worker_optims[i].zero_grad()
						output = worker_models[i].forward(images)
						loss = worker_criterions[i](output, labels)
						loss.backward()
						worker_optims[i].step()

				# Get back the trained model (move to the secure aggregation)
				worker_models[i].move(self.secure_worker)

			# Average all the local models
			model_averaging(worker_models)

		print("Saving model...")
		torch.save(self.model.state_dict(), self.model_weight_path)
		print("Finish training!")

	def train(self):

		'''
		The global model is sent to the worker that contains the data batch
		The local worker trains the model with that data batch
		The local worker sends back the trained model
		This process repeat for all the data for several epochs
		'''

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
		optims = Optims(workers, optim=optimizer)
		criterion = nn.CrossEntropyLoss()

		print("Start training...")

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

				loss.backward()
				opt.step()

				# running_loss += loss.get().item()
				self.model.get()

		print("Saving model...")
		torch.save(self.model.state_dict(), self.model_weight_path)
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