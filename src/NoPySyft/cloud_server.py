import os
import torch
import copy
import numpy as np
from tqdm import tqdm

from torch import nn

from mnist_model import CNNModel
from edge_server_node import EdgeServerNode
from client_node import ClientNode
from data_loader import MNISTDataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

class CloudServer:
	def __init__(self, no_edge_servers, no_clients, num_epochs, batch_size, learning_rate, edge_update, global_update, model_weight_dir):
		self.model = CNNModel().to(device)

		self.num_epochs = num_epochs
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		
		self.edge_servers = self.generate_edge_servers(no_edge_servers)
		self.clients = self.generate_clients(no_clients)

		self.data_loader = MNISTDataLoader(batch_size)

		self.assignment = None

		self.edge_update = edge_update
		self.global_update = global_update

		self.model_weight_dir = model_weight_dir


	def average_models(self, models):
		averaged_model = copy.deepcopy(self.model)

		with torch.no_grad():
			averaged_values = {}
			for name, param in averaged_model.named_parameters():
				averaged_values[name] = nn.Parameter(torch.zeros_like(param.data))

			for model in models:
				for name, param in model.named_parameters():	
					averaged_values[name] += param.data

			for name, param in averaged_model.named_parameters():
				param.data = (averaged_values[name]/len(models))

		return averaged_model


	def send_model_to_edge_servers(self):
		print("Send model to edge servers")
		for edge_server in self.edge_servers:
			edge_server.model = copy.deepcopy(self.model)


	def send_model_to_clients(self):
		for edge_server in self.edge_servers:
			model = copy.deepcopy(edge_server.model)

			for client_id in edge_server.connected_clients:
				client = self.clients[client_id]

				if client.model["model"] == None:
					client.model["model"] = [model]
				else:
					client.model["model"].append(model)


	def generate_edge_servers(self, no_edge_servers):
		print("Generate edge server nodes")
		edge_servers = []
		for i in range(no_edge_servers):
			edge_server = EdgeServerNode(self.model)
			edge_servers.append(edge_server)

		return edge_servers


	def generate_clients(self, no_clients):
		print("Generate client nodes")
		clients = []
		for i in range(no_clients):
			client = ClientNode(self.learning_rate)
			clients.append(client)

		return clients


	def calculate_distance_matrix(self):	
		edge_server_locations = [edge_server.location for edge_server in self.edge_servers]

		distance_matrix = []
		
		for client in self.clients:
			distances = [np.linalg.norm(client.location - edge_server_location) for edge_server_location in edge_server_locations]
			distance_matrix.append(distances)
		
		return np.array(distance_matrix)


	def weight_difference(self, model_A, model_B):
		difference = 0
		with torch.no_grad():
			for param_A, param_B in zip(model_A.parameters(), model_B.parameters()):
				difference += torch.norm(param_A.data - param_B.data)

		return difference


	def calculate_weight_difference_matrix(self):
		difference_matrix = np.zeros((len(self.clients), len(self.clients)))

		for i in tqdm(range(len(self.clients))):
			client_A = self.clients[i]
			model_A = client_A.model["model"]
			
			for j in range(i+1, len(self.clients)):
				
				client_B = self.clients[j]
				model_B = client_B.model["model"]

				difference = self.weight_difference(model_A, model_B)
				difference_matrix[i][j] = difference
				difference_matrix[j][i] = difference

		return difference_matrix


	def random_clients_servers_assign(self):
		clients_per_server = len(self.clients)/len(self.edge_servers)

		assignment = np.zeros((len(self.clients), len(self.edge_servers)), dtype=np.int8)

		for server_id in range(len(self.edge_servers)):
			while assignment.sum(axis=0)[server_id] < clients_per_server:
				client_id = random.randint(0, len(self.clients)-1)
				if np.sum(assignment[client_id]) == 0:
					assignment[client_id][server_id] = 1
					self.edge_servers[server_id].add_client(client_id)

		return assignment


	def shortest_distance_clients_servers_assign(self):
		distance_matrix = self.calculate_distance_matrix()
		distance_matrix = np.transpose(distance_matrix)

		clients_per_server = len(self.clients) / len(self.edge_servers)
		assignment = np.zeros((len(self.clients), len(self.edge_servers)), dtype=np.int8)
		
		for server_id in range(len(self.edge_servers)):
			while assignment.sum(axis=0)[server_id] < clients_per_server:
				nearest_client_id = np.argmin(distance_matrix[server_id])
				if np.sum(assignment[nearest_client_id]) == 0:
					assignment[nearest_client_id][server_id] = 1
					self.edge_servers[server_id].add_client(nearest_client_id)
		
				distance_matrix[server_id][nearest_client_id] = 100

		return assignment


	def k_nearest_edge_servers_assignment(self, k):
		distance_matrix = self.calculate_distance_matrix()

		assignment = np.zeros((len(self.clients), len(self.edge_servers)))

		for client_id in range(len(self.clients)):
			server_indices = np.argpartition(distance_matrix[client_id], k)
			for server_id in server_indices:
				assignment[client_id][server_id] = 1
				self.edge_servers[server_id].add_client(client_id)

		return assignment


	def k_nearest_edge_servers_assignment_fixed_size(self, k):
		distance_matrix = self.calculate_distance_matrix()

		assignment = self.shortest_distance_clients_servers_assign()

		for client_id in range(len(self.clients)):
			server_indices = np.argpartition(distance_matrix[client_id], k)
			for server_id in server_indices:
				if assignment[client_id][server_id] == 0:
					assignment[client_id][server_id] = 1
					self.edge_servers[server_id].add_client(client_id)

				if np.sum(assignment[client_id]) == k:
					break

		return assignment


	def multiple_edges_assignment(self, edge_servers_per_client, alpha, no_local_epochs):
		print("---- Assignment Phase Model Training ----")

		# Send models from edge to nearest workers
		shortest_distance_assignment = self.shortest_distance_clients_servers_assign()

		print("-- Send edge server models to workers --")
		self.send_model_to_clients()

		# Train the local models for a few epochs
		for epoch in range(no_local_epochs):
			print(f"Epoch {epoch+1}/{no_local_epochs}")
			# Train each worker with its own local data
			for i, client in tqdm(enumerate(self.clients)):
				client.train(device)

		# Calculate the distances between workers and edge servers
		print("-- Calculate distance matrix")
		distance_matrix = self.calculate_distance_matrix()

		# Calculate the weight differences between workers
		print("-- Calculate weight difference matrix")
		weight_difference_matrix = self.calculate_weight_difference_matrix()

		# Start the assignment
		print("-- Assign workers to edge server")
		assignment = np.zeros((len(self.clients), len(self.edge_servers)))

		for client_id in range(len(self.clients)):
			cost = alpha*distance_matrix[i][:] + (1-alpha)*np.sum([assignment[i][s]*(1-assignment[j][s])*weight_difference_matrix[i][j] for j in range(i) for s in range(len(self.edge_servers))])
			server_indices = np.argpartition(cost, edge_servers_per_client)
			for server_id in server_indices[:edge_servers_per_client]:
				assignment[client_id][server_id] = 1
				self.edge_servers[server_id].add_client(client_id)

		# Clear models
		for client in self.clients:
			client.model["model"] = None

		return assignment


	def train(self):

		# Load and distribute data to clients
		train_data = self.data_loader.prepare_federated_pathological_non_iid(len(self.clients), train=True)

		print("Distributing data...")
		for client_id, client_data in tqdm(train_data.items()):
			self.clients[client_id].data = client_data

		# Send model to edge server
		self.send_model_to_edge_servers()
		is_updated = True

		# Assigning clients to edge server
		# self.random_clients_servers_assign()
		# self.shortest_distance_clients_servers_assign()
		# self.multiple_edges_assignment(edge_servers_per_client=3, alpha=0.2, no_local_epochs=4)
		self.k_nearest_edge_servers_assignment_fixed_size(k = 3)

		# Train
		print("Start training...")

		accuracy_logs = []
		best_acc = 0

		for epoch in range(self.num_epochs):
			print(f"Epoch {epoch+1}/{self.num_epochs}")

			# Send the edge servers' models to all the workers
			if is_updated:
				print("---- [DELIVER MODEL] Send edge server models to clients ----")
				self.send_model_to_clients()
				is_updated = False 

			# Train each worker
			for i, client in tqdm(enumerate(self.clients)):
				client.train(device)

		
			# Average models at edge servers
			if (epoch+1) % self.edge_update == 0:
				print("---- [UPDATE MODEL] Send local models to edge servers ----")
				is_updated = True
				for edge_server in self.edge_servers:
					models = [self.clients[client_id].model["model"] for client_id in edge_server.connected_clients]
					edge_server.model = self.average_models(models)

				for client in self.clients:
					client.clear_model()

			# Average models at cloud servers
			if (epoch+1) % self.global_update == 0:
				print("---- [UPDATE MODEL] Send edge servers to cloud server ----")
				models = [edge_server.model for edge_server in self.edge_servers if len(edge_server.connected_clients) > 0]
				self.model = self.average_models(models)

				# Validate new model
				accuracy = self.validate(load_weight=False)
				accuracy_logs.append(accuracy)
				if accuracy > best_acc:
					best_acc = accuracy
					self.save_model()
				
				# Send the global model to edge servers
				print("---- [DELIVER MODEL] Send global model to edge servers ----")
				self.send_model_to_edge_servers()
				is_updated = True
				
				for client in self.clients:
					client.clear_model()
				
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

	def save_model(self):
		print("Saving model...")
		if not os.path.exists(self.model_weight_dir):
			os.makedirs(self.model_weight_dir)
		torch.save(self.model.state_dict(), self.model_weight_dir + "/weight.pth")
		print("Model saved!")