import os
import random
from tqdm import tqdm

import numpy as np
import torch
import syft

from torch import nn, optim
import torch.nn.functional as F

from syft.federated.floptimizer import Optims

from mnist_model import CNNModel
from mnist_data_loader import MNIST_DataLoader

hook = syft.TorchHook(torch)

seed = 1

torch.manual_seed(seed)
random.seed(seed)

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
		self.init_worker_locations()

		self.edge_servers = self.init_edge_servers(num_edge_servers)

		self.secure_worker = self.init_secure_worker()

		self.workers_per_server = workers_per_server

		self.data_loader = MNIST_DataLoader(batch_size, workers=[worker["instance"] for worker in self.workers])

		self.edge_update = edge_update
		self.global_update = global_update

		self.iid = iid

	
	def init_workers(self, num_workers):
		workers = []
		for i in range(num_workers):
			worker = {}
			worker["instance"] = syft.VirtualWorker(hook, id=f"worker_{i}")
			worker["model"] = None
			worker["optim"] = None
			worker["criterion"] = None
			worker["loss"] = None 

			workers.append(worker)
		return workers

	
	def init_edge_servers(self, num_edge_servers):
		return [syft.VirtualWorker(hook, id=f"server_{i}") for i in range(num_edge_servers)]

	
	def init_secure_worker(self):
		return syft.VirtualWorker(hook, id="secure_worker")


	def init_worker_locations(self):
		location_ranges = [0,1,2,3,4]
		distributions = [0.1, 0.15, 0.2, 0.25, 0.3]

		for worker in self.workers:
			index = np.random.choice(location_ranges, 1, replace=False, p=distributions)[0]
			start = 0.2 * index
			end = start + 0.2

			worker["location"] = (random.uniform(start, end), random.uniform(start, end))


	def calculate_distance_matrix(self):	
		server_locations = [np.array((random.random(), random.random())) for i in range(len(self.edge_servers))]

		distance_matrix = []
		
		for worker in self.workers:
			distances = [np.linalg.norm(worker["location"] - server_location) for server_location in server_locations]
			distance_matrix.append(distances)
		
		return np.array(distance_matrix)


	def weight_difference(self, model_A, model_B):
		difference = 0
		with torch.no_grad():
			for param_A, param_B in zip(model_A.parameters(), model_B.parameters()):
				difference += torch.norm(param_A.data - param_B.data)

		return difference


	def calculate_weight_difference_matrix(self):
		difference_matrix = np.zeros((len(self.workers), len(self.workers)))

		for i in tqdm(range(len(self.workers))):
			worker_A = self.workers[i]
			model_A = worker_A["model"].get_()
			
			for j in range(i+1, len(self.workers)):
				
				worker_B = self.workers[j]
				model_B = worker_B["model"].get_()

				difference = self.weight_difference(model_A, model_B)
				difference_matrix[i][j] = difference
				difference_matrix[j][i] = difference

				worker_B["model"] = model_B.send(worker_B["instance"])

			worker_A["model"] = model_A.send(worker_A["instance"])

		return difference_matrix

	
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


	def shortest_distance_workers_servers_assign(self):
		distance_matrix = self.calculate_distance_matrix()
		distance_matrix = np.transpose(distance_matrix)

		workers_per_server = len(self.workers) / len(self.edge_servers)
		assignment = np.zeros((len(self.workers), len(self.edge_servers)), dtype=np.int8)
		
		for server_id in range(len(self.edge_servers)):
			while assignment.sum(axis=0)[server_id] < workers_per_server:
				nearest_worker_id = np.argmin(distance_matrix[server_id])
				if np.sum(assignment[nearest_worker_id]) == 0:
					assignment[nearest_worker_id][server_id] = 1
		
				distance_matrix[server_id][nearest_worker_id] = 100

		return assignment


	def multiple_edges_assignment(self, edge_server_models, edge_servers_per_worker, alpha, train_data, no_epochs_local):
		print("---- Assignment Phase Model Training ----")

		# Send models from edge to nearest workers
		shortest_distance_assignment = self.shortest_distance_workers_servers_assign()

		print("-- Send edge server models to workers --")
		for server_id in tqdm(range(len(self.edge_servers))):
			connected_workers_ids = np.where(shortest_distance_assignment[:,server_id] == 1)[0]
			if len(connected_workers_ids) > 0:
				self.send_model_to_workers(source=edge_server_models[server_id], worker_ids=connected_workers_ids)

		# Train the local models for a few epochs
		for epoch in range(no_epochs_local):
			print(f"Epoch {epoch+1}/{no_epochs_local}")
			# Train each worker with its own local data
			for worker_id, worker in tqdm(enumerate(self.workers)):

				if isinstance(worker["model"], list):
					if len(worker["model"]) > 1:
						average_model = self.average_models(worker["model"], local=True)
						worker["model"] = average_model.send(worker["instance"])
					else:
						worker["model"] = worker["model"][0]

				worker["optim"] = optim.SGD(worker["model"].parameters(), lr=self.lr)
				worker["criterion"] = nn.CrossEntropyLoss() 

				# Train worker's model
				for batch_idx, (images, labels) in enumerate(train_data[worker_id]):
					
					images, labels = images.to(device), labels.to(device)

					if (batch_idx+1)%100==0:
						print(f"Processed {batch_idx+1}/{len(train_data[worker_id])} batches")

					worker["optim"].zero_grad()
					output = worker["model"].forward(images)
					worker["loss"] = worker["criterion"](output, labels)
					worker["loss"].backward()
					worker["optim"].step()

		# Calculate the distances between workers and edge servers
		print("-- Calculate distance matrix")
		distance_matrix = self.calculate_distance_matrix()

		# Calculate the weight differences between workers
		print("-- Calculate weight difference matrix")
		weight_difference_matrix = self.calculate_weight_difference_matrix()

		# Start the assignment
		print("-- Assign workers to edge server")
		assignment = np.zeros((len(self.workers), len(self.edge_servers)))

		for i, worker in enumerate(self.workers):
			cost = alpha*distance_matrix[i][:] + (1-alpha)*np.sum([assignment[i][s]*(1-assignment[j][s])*weight_difference_matrix[i][j] for j in range(i) for s in range(len(self.edge_servers))])
			server_indices = np.argpartition(cost, edge_servers_per_worker)
			for server_id in server_indices[:edge_servers_per_worker]:
				assignment[i][server_id] = 1

		# Clear models
		for worker in self.workers:
			worker["model"] = None

		return assignment


	def random_workers_servers_assign(self):
		workers_per_server = len(self.workers)/len(self.edge_servers)

		assignment = np.zeros((len(self.workers), len(self.edge_servers)))

		for server_id in range(len(self.edge_servers)):
			while assignment.sum(axis=0)[server_id] < workers_per_server:
				worker_id = random.randint(0, len(self.workers)-1)
				if np.sum(assignment[worker_id]) == 0:
					assignment[worker_id][server_id] = 1

		return assignment


	def send_model_to_workers(self, source, worker_ids):
		for worker_id in worker_ids:
			worker = self.workers[worker_id]

			model_clone = source.copy().send(worker["instance"])
			
			if worker["model"] == None:
				worker["model"] = [model_clone]
			else:
				worker["model"].append(model_clone)


	def average_models(self, models, local):
		# Create a model to hold the data
		averaged_model = self.model.copy()

		# Average the models
		with torch.no_grad():
			averaged_values = {}
			for name, param in averaged_model.named_parameters():
				averaged_values[name] = nn.Parameter(torch.zeros_like(param.data))

			if local == True:
				for i in range(len(models)):
					model_location = models[i].location
					model = models[i].get_()

					for name, param in model.named_parameters():
						averaged_values[name] += param.data

					models[i] = model.send(model_location)
			else:
				for model in models:
					for name, param in model.named_parameters():	
						averaged_values[name] += param.data

			for name, param in averaged_model.named_parameters():
				param.data = (averaged_values[name]/len(models))

			return averaged_model


	def train(self):		

		if self.iid == True:
			print("---- Train in Federated Hierachical IID Mode ----")
			train_data = self.data_loader.prepare_federated_iid_data_parallel(train=True)
		else:
			print("---- Train in Federated Hierarchical Non-IID Mode ----")
			train_data = self.data_loader.prepare_federated_pathological_non_iid(train=True)
		
		# Send the global model to each edge server
		print("---- Send global model to edge servers ----")
		edge_server_models = [self.model.copy()] * len(self.edge_servers)
		is_updated = True

		# Assign workers to edge servers

		# assignment = self.random_workers_servers_assign()
		# assignment = self.shortest_distance_workers_servers_assign()
		assignment = self.multiple_edges_assignment(edge_server_models=edge_server_models, edge_servers_per_worker=3, alpha=0.2, train_data=train_data, no_epochs_local=4)


		print("Start training...")

		accuracy_logs = []
		best_acc = 0
		
		for epoch in range(self.num_epochs):
			print(f"Epoch {epoch+1}/{self.num_epochs}")

			# Send the edge servers' models to all the workers
			if is_updated:
				print("---- Send edge server models to workers ----")
				for server_id in tqdm(range(len(self.edge_servers))):
					connected_workers_ids = np.where(assignment[:,server_id] == 1)[0]
					if len(connected_workers_ids) > 0:
						self.send_model_to_workers(source=edge_server_models[server_id], worker_ids=connected_workers_ids)
				is_updated = False 

			# Train each worker
			for worker_id, worker in tqdm(enumerate(self.workers)):
				# If there are multiple models per worker, average them
				if isinstance(worker["model"], list):
					if len(worker["model"]) > 1:
						average_model = self.average_models(worker["model"], local=True)
						worker["model"] = average_model.send(worker["instance"])
					else:
						worker["model"] = worker["model"][0]


				worker["optim"] = optim.SGD(worker["model"].parameters(), lr=self.lr)
				worker["criterion"] = nn.CrossEntropyLoss() 


				for batch_idx, (images, labels) in enumerate(train_data[worker_id]):
						
						images, labels = images.to(device), labels.to(device)

						if (batch_idx+1)%100==0:
							print(f"Processed {batch_idx+1}/{len(train_data[worker_id])} batches")

						worker["optim"].zero_grad()
						output = worker["model"].forward(images)
						loss = worker["criterion"](output, labels)
						loss.backward()
						worker["optim"].step()

			if (epoch+1) % self.edge_update == 0:
				print("---- Send local models to edge servers ----")
				is_updated = True
				for server_id, edge_server in enumerate(self.edge_servers):
					connected_workers_ids = np.where(assignment[:,server_id] == 1)[0]
					models = [self.workers[worker_id]["model"] for worker_id in connected_workers_ids]

					edge_server_models[server_id] = self.average_models(models, local=True)

				for worker in self.workers:
					worker["model"] = None

					#TODO: clear PySyft Tensor models

			
			if (epoch+1) % self.global_update == 0:
				print("---- Send edge servers to cloud server ----")
				models = [model for server_id, model in enumerate(edge_server_models) if assignment.sum(axis=0)[server_id] > 0]
				self.model = self.average_models(edge_server_models, local=False)

				# Validate new model
				accuracy = self.validate(load_weight=False)
				accuracy_logs.append(accuracy)
				if accuracy > best_acc:
					best_acc = accuracy
					self.save_model()
				
				# Send the global model to edge servers
				print("--Send global model to edge servers--")
				edge_server_models = [self.model.copy()] * len(self.edge_servers)
				is_updated = True

				for worker in self.workers:
					worker["model"] = None
				
		print("Finish training!")


	def validate(self, load_weight=False):
		print("-------------------------------------------")
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
		print("-------------------------------------------")

		return accuracy
