import os

import torch
import syft

from torch import nn, optim
import torch.nn.functional as F

from syft.federated.floptimizer import Optims

from mnist_model import CNNModel
from mnist_data_loader import MNIST_DataLoader

hook = syft.TorchHook(torch)

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FederatedTrainer:
	def __init__(self, batch_size, lr, num_rounds, num_epochs, model_weight_dir, num_workers, iid, parallel):
		self.model = CNNModel().to(device)
		
		self.lr = lr
		self.batch_size = batch_size
		self.num_rounds = num_rounds
		self.num_epochs = num_epochs
		self.model_weight_dir = model_weight_dir

		self.workers = self.init_workers(num_workers)
		self.secure_worker = self.init_secure_worker()

		self.data_loader = MNIST_DataLoader(batch_size, workers=self.workers)

		self.iid = iid
		self.parallel = parallel

	
	def init_workers(self, num_workers):
		return [syft.VirtualWorker(hook, id=f"worker_{i}") for i in range(num_workers)]

	
	def init_secure_worker(self):
		return syft.VirtualWorker(hook, id="secure_worker")

	
	def load_data(self, train):
		transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
		data = datasets.MNIST('../data', train=train, download=True, transform=transform)
		
		return data


	def save_model(self):
		print("Saving model...")
		if not os.path.exists(self.model_weight_dir):
			os.makedirs(self.model_weight_dir)
		torch.save(self.model.state_dict(), self.model_weight_dir + "/weight.pth")
		print("Model saved!")

	
	def train(self):
		if self.parallel:
			self.train_parallel()
		else:
			self.train_sequential()


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


		def send_model(source, receivers):
			worker_models = []
			worker_optims = []
			worker_criterions = []
			worker_losses = []

			for i in range(len(receivers)):
				model_clone = source.copy().send(receivers[i])

				worker_models.append(model_clone)
				worker_optims.append(optim.SGD(model_clone.parameters(), lr=self.lr))
				worker_criterions.append(nn.CrossEntropyLoss())

			return worker_models, worker_optims, worker_criterions, worker_losses

		if self.iid == True:
			print("Train in Federated IID Mode")
			train_data = self.data_loader.prepare_federated_iid_data_parallel(train=True)
		else:
			print("Train in Federated Non-IID Mode")
			train_data = self.data_loader.prepare_federated_pathological_non_iid(train=True)
			# train_data = self.data_loader.prepare_federated_non_iid_data_parallel(train=True)

		print("Start training...")

		accuracy_logs = []

		best_acc = 0

		for round_iter in range(self.num_rounds):
			worker_models, worker_optims, worker_criterions, worker_losses = send_model(source=self.model, receivers=self.workers)

			# Train each worker with its own local data
			for i, worker_model in enumerate(worker_models):

				# Train worker's model
				for epoch in range(self.num_epochs):
					print(f"Round {round_iter+1}/{self.num_rounds} - Worker {i+1}/{len(self.workers)} - Epoch {epoch+1}/{self.num_epochs}")
					for batch_idx, (images, labels) in enumerate(train_data[i]):
						
						images, labels = images.to(device), labels.to(device)

						if (batch_idx+1)%100==0:
							print(f"Processed {batch_idx+1}/{len(train_data[i])} batches")

						worker_optims[i].zero_grad()
						output = worker_model.forward(images)
						loss = worker_criterions[i](output, labels)
						loss.backward()
						worker_optims[i].step()

				# Get back the trained model (move to the secure aggregation)
				worker_model.move(self.secure_worker)

			# Average all the local models
			model_averaging(worker_models)
			accuracy = self.validate(load_weight=False)

			if accuracy > best_acc:
				best_acc = accuracy
				self.save_model()

		print("Finish training!")

	def train_sequential(self):

		'''
		The global model is sent to the worker that contains the data batch
		The local worker trains the model with that data batch
		The local worker sends back the trained model
		This process repeat for all the data for several epochs
		'''

		if self.iid == True:
			print("Train in Federated IID Mode")
			train_data = self.data_loader.prepare_federated_iid_data_sequential(train=True)
		else:
			print("Train in Federated Non-IID Mode")
			train_data = self.data_loader.prepare_federated_non_iid_data_sequential(train=True)
	
		workers = list()
		for worker in self.workers:
			workers.append(worker.id)
		
		optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
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

				if (batch_idx+1) % 100 == 0: 
					print("Processed {}/{} batches".format(batch_idx+1, len(train_data)))

				optimizer.zero_grad()

				images, labels = images.to(device), labels.to(device)

				output = self.model(images)
				
				loss = criterion(output, labels)

				loss.backward()
				opt.step()

				# running_loss += loss.get().item()
				self.model.get()

			self.validate(load_weight=False)

			if accuracy > best_acc:
				best_acc = accuracy
				self.save_model()


	def validate(self, load_weight=False):
		print("-----------------------------------------")
		print("Start validating...")

		if load_weight == True:
			self.model.load_state_dict(torch.load(self.model_weight_dir + "/weight.pth"))

		self.model.eval()
		corrects = 0

		test_data = self.data_loader.test_data
		
		with torch.no_grad():
			for batch_idx, (images, labels) in enumerate(test_data):
				images, labels = images.to(device), labels.to(device)
				output = self.model(images)
				pred = output.argmax(dim=1)
				corrects += pred.eq(labels.view_as(pred)).sum().item()


		total_test = len(test_data)*self.batch_size
		accuracy = 100*corrects/total_test

		print("Number of corrects: {}/{}".format(corrects, len(test_data)*self.batch_size))
		print("Accuracy: {}%".format(accuracy))
		print("-----------------------------------------")

		return accuracy