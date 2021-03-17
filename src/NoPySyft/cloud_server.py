import os
import torch
import numpy as np
from tqdm import tqdm
import random

from torch import nn

from mnist_model import CNNModel
from edge_server_node import EdgeServerNode
from client_node import ClientNode
from data_loader import MNISTDataLoader
from client_assignment import ClientAssignment

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

		self.edge_update = edge_update
		self.global_update = global_update

		self.model_weight_dir = model_weight_dir


	def copy_model(self, source_model):
		model_copy = type(source_model)()
		model_copy.load_state_dict(source_model.state_dict())

		return model_copy


	def average_models(self, models):
		averaged_model = type(self.model)().to(device)

		with torch.no_grad():
			averaged_dict = averaged_model.state_dict()
			for k in averaged_dict.keys():
				averaged_dict[k] = torch.stack([models[i].state_dict()[k].float() for i in range(len(models))], 0).mean(0)
			
			averaged_model.load_state_dict(averaged_dict)

		return averaged_model


	def send_cloud_model_to_edge_servers(self):
		for edge_server in self.edge_servers:
			edge_server.model = self.copy_model(self.model).to(device)


	def send_cloud_model_to_clients(self):
		for client in self.clients:
			client.model["model"] = [self.copy_model(self.model).to(device)]


	def send_edge_models_to_clients(self):
		for edge_server in self.edge_servers:
			for client_id in edge_server.connected_clients:
				client = self.clients[client_id]
				
				model = self.copy_model(edge_server.model).to(device)
				
				if client.model["model"] == None:
					client.model["model"] = [model]
				else:
					client.model["model"].append(model)


	def generate_edge_servers(self, no_edge_servers):
		print("Generate edge server nodes")
		edge_servers = []
		for i in range(no_edge_servers):
			edge_server = EdgeServerNode()
			edge_servers.append(edge_server)

		return edge_servers


	def generate_clients(self, no_clients):
		print("Generate client nodes")
		clients = []
		for i in range(no_clients):
			client = ClientNode(self.learning_rate)
			clients.append(client)

		return clients


	def calculate_assignment_cost(self, assignment, distance_matrix, weight_difference_matrix, alpha):
		cost = 0
		for client_cost in range(len(self.clients)):
			client_cost = alpha*distance_matrix[client_id][:] + (1-alpha)*np.sum([assignment[client_id][s]*(1-assignment[j][s])*weight_difference_matrix[client_id][j] for j in range(client_id) for s in range(len(self.edge_servers))])
			cost+= client_cost
		return cost


	def distribute_data_to_clients(self, train_data):
		client_ids = list(range(len(self.clients)))
		for _, data in tqdm(train_data.items()):
			selected_client = random.choice(client_ids)
			self.clients[selected_client].data = data
			client_ids.remove(selected_client)


	def train(self):

		# Load and distribute data to clients
		train_data = self.data_loader.prepare_federated_pathological_non_iid(len(self.clients))

		print("---- [DISTRIBUTE DATA] Send data to clients ----")
		for client_id, client_data in train_data.items():
			self.clients[client_id].data = client_data

		# Send model to edge server
		print("---- [DELIVER MODEL] Send global model to clients ----")
		self.send_cloud_model_to_clients()
		edge_updated = False

		# Assigning clients to edge server
		# client_assignment = ClientAssignment().random_clients_servers_assign(self.clients, self.edge_servers)
		# client_assignment = ClientAssignment().shortest_distance_clients_servers_assign(self.clients, self.edge_servers)
		client_assignment = ClientAssignment().multiple_edges_assignment(self.clients, self.edge_servers, k=3, alpha=0.0, no_local_epochs=5)
		# client_assignment = ClientAssignment().random_multiple_edges_assignment(self.clients, self.edge_servers, k=3)
		# client_assignment = ClientAssignment().k_nearest_edge_servers_assignment_fixed_size(self.clients, self.edge_servers, k = 3)

		np.save("client_assignment.npy", client_assignment)

		# Train
		print("Start training...")

		print("---- [DELIVER MODEL] Send global model to clients ----")
		self.send_cloud_model_to_clients()

		accuracy_logs = []
		best_acc = 0

		for epoch in range(self.num_epochs):
			print(f"Epoch {epoch+1}/{self.num_epochs}")

			# Send the edge servers' models to all the workers
			if edge_updated:
				print("---- [DELIVER MODEL] Send edge server models to clients ----")
				self.send_edge_models_to_clients()
				edge_updated = False 

			# Train each worker
			for i, client in tqdm(enumerate(self.clients)):
				client.train()
		
			# Average models at edge servers
			if (epoch+1) % self.edge_update == 0:
				print("---- [UPDATE MODEL] Send local models to edge servers ----")
				edge_updated = True
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

				np.save("accuracy_logs.npy", accuracy_logs)		
				
				# Clear models
				for client in self.clients:
					client.clear_model()

				edge_updated = False
				for edge_server in self.edge_servers:
					edge_server.clear_model()

				# Send the global model to edge servers
				print("---- [DELIVER MODEL] Send global model to clients ----")
				self.send_cloud_model_to_clients()
		
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
		if not os.path.exists(self.model_weight_dir):
			os.makedirs(self.model_weight_dir)
		torch.save(self.model.state_dict(), self.model_weight_dir + "/weight.pth")
		print("Model saved!")