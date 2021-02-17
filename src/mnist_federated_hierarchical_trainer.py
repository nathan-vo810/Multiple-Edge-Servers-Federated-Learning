import os
import random

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

class FederatedHierachicalTrainer:
	def __init__(self, batch_size, lr, num_rounds, num_epochs, model_weight_dir, num_workers, num_edge_servers, workers_per_server, edge_update, global_update, iid):
		self.model = CNNModel().to(device)
		
		self.lr = lr
		self.batch_size = batch_size
		self.num_rounds = num_rounds
		self.num_epochs = num_epochs
		self.model_weight_dir = model_weight_dir

		self.workers = self.init_workers(num_workers)
		self.edge_servers = self.init_edge_servers(num_edge_servers)
		self.secure_worker = self.init_secure_worker()

		self.workers_per_server = workers_per_server

		self.data_loader = MNIST_DataLoader(batch_size, workers=self.workers)

		self.edge_update = edge_update
		self.global_update = global_update

		self.iid = iid

	
	def init_workers(self, num_workers):
		return [syft.VirtualWorker(hook, id=f"worker_{i}") for i in range(num_workers)]

	
	def init_edge_servers(self, num_edge_servers):
		return [syft.VirtualWorker(hook, id=f"server_{i}") for i in range(num_edge_servers)]

	
	def init_secure_worker(self):
		return syft.VirtualWorker(hook, id="secure_worker")


	def init_distances(self):
		distances = np.array()
		return distances

	
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


	def random_workers_servers_assign(self):
		workers_assigned = [False] * len(self.workers)
		workers_per_server = len(self.workers)/len(self.edge_servers)

		assignment = {}

		for edge_server in self.edge_servers:
			assignment[edge_server] = []
			while len(assignment[edge_server]) < workers_per_server:
				worker_id = random.randint(0, len(self.workers)-1)
				if workers_assigned[worker_id] == False:
					workers_assigned[worker_id] = True
					assignment[edge_server].append(self.workers[worker_id])

		return assignment

	def train(self):

		def send_model(source, receivers):
			worker_models = []
			worker_optims = []
			worker_criterions = []
			worker_losses = []

			for receiver in receivers:
				model_clone = source.copy().send(receiver)

				worker_models.append(model_clone)
				worker_optims.append(optim.SGD(model_clone.parameters(), lr=self.lr))
				worker_criterions.append(nn.CrossEntropyLoss())

			return worker_models, worker_optims, worker_criterions, worker_losses

		def edge_averaging(local_models, target_model):
			with torch.no_grad():
				averaged_values = {}
				for name, param in target_model.named_parameters():
					averaged_values[name] = nn.Parameter(torch.zeros_like(param.data))

				for local_model in local_models:
					for name, local_param in local_model.named_parameters():
						averaged_values[name] += local_param.get().data

				for name, param in target_model.named_parameters():
					param.data = (averaged_values[name]/len(local_models))

		def global_averaging(local_models, target_model):
			return

		if self.iid == True:
			print("Train in Federated Hierachical IID Mode")
			train_data = self.data_loader.prepare_federated_iid_data_parallel(train=True)
		else:
			print("Train in Federated Hierarchical Non-IID Mode")
			train_data = self.data_loader.prepare_federated_pathological_non_iid(train=True)

		print("Start training...")

		accuracy_logs = []
		best_acc = 0

		# Assign workers to edge servers
		assignment = self.random_workers_servers_assign()

		# Send the global model to each edge server
		print("--Send global model to edge servers--")
		edge_server_models = [self.model] * len(self.edge_servers)
		is_updated = [True]*len(self.edge_servers)
		
		for epoch in range(self.num_epochs):
			
			# Train each edge server
			for k, edge_server in enumerate(self.edge_servers):
				# Send the edge model to the connected workers
				if is_updated[k]:
					print("--Send edge model to local workers--")
					worker_models, worker_optims, worker_criterions, worker_losses = send_model(source=edge_server_models[k], receivers=assignment[edge_server])
					is_updated[k] = False

				# Train each worker with its own local data
				for i, worker_model in enumerate(worker_models):
					print(worker_model)
					worker_id = int(assignment[edge_server][i].id.split("_")[1])

					# Train worker's model
					print(f"Epoch {epoch+1}/{self.num_epochs} - Worker {i+1}/{self.workers_per_server} - ID {worker_id}")
					for batch_idx, (images, labels) in enumerate(train_data[worker_id]):
						
						images, labels = images.to(device), labels.to(device)

						if (batch_idx+1)%100==0:
							print(f"Processed {batch_idx+1}/{len(train_data[i])} batches")

						worker_optims[i].zero_grad()
						output = worker_model.forward(images)
						loss = worker_criterions[i](output, labels)
						loss.backward()
						worker_optims[i].step()

			# After every E epoch, average the models at each edge server
			if (epoch+1) % self.edge_update == 0:
				print("--Edge Model Average--")
				for k, edge_server in enumerate(self.edge_servers):
					# List of connected workers models
					local_models = [worker_models[worker.id.split("_")[1]] for worker in assignment[edge_servers]]
					
					# Move local models to secure worker for averaging
					for model in local_models:
						model.move(self.secure_worker)

					# Average all the local models of the edge server
					model_averaging(local_models, target_model=edge_server_models[k])					

					# Signal that new model is available
					is_updated[k] = True
				print("--Done--")

			# After every G epoch average the models at the cloud
			if (epoch+1) % self.global_update == 0:
				print("--Global Model Average--")
				for edge_server_model in edge_server_models:
					edge_server_model.move(self.secure_worker)

				model_averaging(edge_server_models, target_model=self.model)

				print("--Done--")

				accuracy = self.validate(load_weight=False)
				accuracy_logs.append(accuracy)
				if accuracy > best_acc:
					best_acc = accuracy
					self.save_model()
				
				# Send the global model to edge servers
				print("--Send global model to edge servers--")
				edge_server_models = [self.model] * len(self.edge_servers)
				is_updated = [True] * len(self.edge_servers)
					
		print("Finish training!")


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
