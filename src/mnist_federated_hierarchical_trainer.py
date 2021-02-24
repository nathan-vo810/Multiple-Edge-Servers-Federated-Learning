import os
import random

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
			worker["losses"] = None 

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
		
		return distance_matrix


	def weight_difference(self, model_A, model_B):
		difference = 0
		with torch.no_grads():
			for param_A, param_B in zip(model_A.parameters(), model_B.parameters()):
			difference += torch.norm(param_A.get_().data - param_B.get_().data)

		return difference


	def calculate_weight_difference_matrix(self):
		difference_matrix = []

		for i, worker_A in enumerate(self.workers):
			difference_matrix[i] = []
			for j, worker_B in enumerate(self.workers):
				if i == j:
					difference_matrix[i].append(0)
				else:
					model_A = worker_A["model"]
					model_B = worker_B["model"]

					difference = weight_difference(model_A, model_B)
					difference_matrix.append(difference)

					worker_A["model"] = model_A.send(worker_A["instance"])
					worker_B["model"] = model_B.send(worker_B["instance"])

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


	


	def multiple_edges_assignment(edge_servers_per_worker, alpha):
		distance_matrix = self.calculate_distance_matrix()
		weight_difference_matrix = self.calculate_weight_difference_matrix()

		assignment = {}

		for edge_server in self.edge_servers:
			assignment[edge_server] = []





	def shortest_distance_workers_servers_assign(self):
		distance_matrix = self.calculate_distance_matrix()
		distance_matrix = np.transpose(distance_matrix)

		workers_per_server = len(self.workers) / len(self.edge_servers)
		workers_assigned = [False] * len(self.workers)

		assignment = {}
		for i, edge_server in enumerate(self.edge_servers):
			assignment[edge_server] = []
			while len(assignment[edge_server]) < workers_per_server:
				nearest_worker_id = np.argmin(distance_matrix[i], axis=0)
				if workers_assigned[nearest_worker_id] == False:
					workers_assigned[nearest_worker_id] = True
					assignment[edge_server].append(nearest_worker_id)
		
				distance_matrix[i][nearest_worker_id] = 10

		return assignment


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
					# assignment[edge_server].append(self.workers[worker_id])
					assignment[edge_server].append(worker_id)

		return assignment


	def location_workers_servers_assign(self):
		server_locations = self.init_server_locations()

		distances = calculate_distances(server_locations, self.workers)

	def send_model_to_workers(self, source, worker_ids):
		worker_models = []
		worker_optims = []
		worker_criterions = []
		worker_losses = []

		for worker_id in worker_ids:
			worker = self.workers[worker_id]

			model_clone = source.copy().send(worker["instance"])
			worker["model"] = model_clone

			worker["optim"] = optim.SGD(model_clone.parameters(), lr=self.lr)
			worker["criterion"] = nn.CrossEntropyLoss() 

	def model_averaging(self, local_models, target_model, edge_averaging):
		with torch.no_grad():
			averaged_values = {}
			for name, param in target_model.named_parameters():
				averaged_values[name] = nn.Parameter(torch.zeros_like(param.data))

			for local_model in local_models:
				for name, local_param in local_model.named_parameters():	
					if edge_averaging == True:
						averaged_values[name] += local_param.get().data
					else:
						averaged_values[name] += local_param.data
					
			for name, param in target_model.named_parameters():
				param.data = (averaged_values[name]/len(local_models))

	def assign(self):


	def train(self):

		

		# if self.iid == True:
		# 	print("Train in Federated Hierachical IID Mode")
		# 	train_data = self.data_loader.prepare_federated_iid_data_parallel(train=True)
		# else:
		# 	print("Train in Federated Hierarchical Non-IID Mode")
		# 	train_data = self.data_loader.prepare_federated_pathological_non_iid(train=True)

		print("Start training...")

		accuracy_logs = []
		best_acc = 0

		# Assign workers to edge servers

		# assignment = self.random_workers_servers_assign()
		assignment = self.shortest_distance_workers_servers_assign()

		# Send the global model to each edge server
		print("--Send global model to edge servers--")
		edge_server_models = [self.model.copy()] * len(self.edge_servers)
		is_updated = [True]*len(self.edge_servers)

		print(edge_server_models)
		
		for epoch in range(self.num_epochs):
			print(f"Epoch {epoch+1}/{self.num_epochs}")

			# Train each edge server
			for k, edge_server in enumerate(self.edge_servers):
				print(f"Edge Server {k+1}/{len(self.edge_servers)}")

				# If there is a new model, send the edge model to the connected workers
				if is_updated[k]:
					print("--Send edge model to local workers--")
					self.send_model_to_workers(source=edge_server_models[k], worker_ids=assignment[edge_server])
					is_updated[k] = False

				# Train each worker with its own local data
				for i in range(self.workers_per_server):

					worker_id = assignment[edge_server][i]
					worker = self.workers[worker_id]

					# Train worker's model
					print(f"Worker {i+1}/{self.workers_per_server} - ID {worker_id}")
					for batch_idx, (images, labels) in enumerate(train_data[worker_id]):
						
						images, labels = images.to(device), labels.to(device)

						if (batch_idx+1)%100==0:
							print(f"Processed {batch_idx+1}/{len(train_data[i])} batches")

						worker["optim"].zero_grad()
						output = worker["model"].forward(images)
						loss = worker["criterion"](output, labels)
						loss.backward()
						worker["optim"].step()

			# After every E epoch, average the models at each edge server
			if (epoch+1) % self.edge_update == 0:
				print("--Edge Models Average--")
				for k, edge_server in enumerate(self.edge_servers):
					# List of connected workers models
					local_models = [self.workers[worker_id]["model"] for worker_id in assignment[edge_server]]
					
					# Move local models to secure worker for averaging
					for model in local_models:
						model.move(self.secure_worker)

					# Average all the local models of the edge server
					self.model_averaging(local_models, target_model=edge_server_models[k], edge_averaging=True)					

					# Signal that new model is available
					is_updated[k] = True

			# After every G epoch average the edge models at the cloud
			if (epoch+1) % self.global_update == 0:
				print("--Global Model Average--")
				# for i in range(len(edge_server_models)):
				# 	edge_server_models[i] = edge_server_models[i].send(self.secure_worker)

				self.model_averaging(edge_server_models, target_model=self.model, edge_averaging=False)

				accuracy = self.validate(load_weight=False)
				accuracy_logs.append(accuracy)
				if accuracy > best_acc:
					best_acc = accuracy
					self.save_model()
				
				# Send the global model to edge servers
				print("--Send global model to edge servers--")
				edge_server_models = [self.model.copy()] * len(self.edge_servers)
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
