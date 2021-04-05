import os
import torch
import numpy as np
from tqdm import tqdm
import random

from torch import nn

from mnist_model import CNNModel, NNModel
from edge_server_node import EdgeServerNode
from client_node import ClientNode
from data_loader import MNISTDataLoader
from client_assignment import ClientAssignment
from edge_server_assignment import EdgeServerAssignment

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(2)


class CloudServer:
	def __init__(self, no_edge_servers, no_clients, num_epochs, batch_size, learning_rate, edge_update, global_update, edges_exchange, edges_param, model_weight_dir):
		self.model = NNModel().to(device)

		self.num_epochs = num_epochs
		self.learning_rate = float(learning_rate)
		self.batch_size = batch_size
		
		self.edge_servers = self.generate_edge_servers(no_edge_servers)
		self.clients = self.generate_clients(no_clients)

		self.data_loader = MNISTDataLoader(batch_size)

		self.edge_update = edge_update
		self.global_update = global_update

		self.edges_exchange = edges_exchange
		self.edges_param = edges_param

		self.model_weight_dir = model_weight_dir


	def copy_model(self, source_model):
		model_copy = type(source_model)()
		model_copy.load_state_dict(source_model.state_dict())

		return model_copy


	def average_models(self, models):
		averaged_model = self.copy_model(self.model).to(device)

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


	def create_incidence_matrix(self, adjacency_matrix):
		num_vertices = adjacency_matrix.shape[0]
		num_edges = int(np.sum(adjacency_matrix)/2)

		incidence_matrix = np.zeros((num_vertices, num_edges))

		edge_num = 0
		for i in range(num_vertices):
			for j in range(i+1, num_vertices):
				if adjacency_matrix[i][j] == 1:
					incidence_matrix[i][edge_num] = 1 	# outgoing vertice get 1
					incidence_matrix[j][edge_num] = -1 	# incoming vertice get -1
					edge_num += 1
		
		return incidence_matrix


	def calculate_constant_edge_weights(self, assignment):
		A = self.create_incidence_matrix(assignment)
		L = A.dot(np.transpose(A))

		eigenvalues = np.linalg.eig(L)[0]
		eigenvalues.sort()

		alpha = 2 / (eigenvalues[1] + eigenvalues[-1]) # sum of the largest and the next to smallest

		return alpha


	def save_training_loss(self, loss_dir):
		np.save(f"{loss_dir}/{self.client_id}_loss.npy", np.array(self.model["loss"]))


	def sub_models(self, model_A, model_B):
		model_A_weight = model_A.state_dict()
		model_B_weight = model_B.state_dict()

		weight_difference = type(self.model)().state_dict()
		for layer in weight_difference.keys():
			weight_difference[layer] = model_A_weight[layer] - model_B_weight[layer]

		return weight_difference


	def sum_diffs(self, diff_A, diff_B):
		for layer in diff_A.keys():
			diff_A[layer] += diff_B[layer]

		return diff_A


	def sum_dict(self, d):
		s = 0.0
		for layer in d.keys():
			s += torch.sum(d[layer])

		return s


	def distributed_edges_average(self, alpha):
		new_weights = []

		for edge_server in self.edge_servers:
			total_diff = edge_server.model.state_dict()
			for layer in total_diff.keys():
				total_diff[layer] = 0.0

			for server_id in edge_server.neighbor_servers:
				neighbor_server = self.edge_servers[server_id]

				if len(neighbor_server.connected_clients) > 0:
					diff = self.sub_models(neighbor_server.model, edge_server.model)
					total_diff = self.sum_diffs(total_diff, diff)
			
			for layer in total_diff.keys():
				total_diff[layer] *= alpha

			new_weight = edge_server.model.state_dict()
			for layer in new_weight.keys():
				new_weight[layer] += total_diff[layer]

			new_weights.append(new_weight)

		for i, edge_server in enumerate(self.edge_servers):
			if len(edge_server.connected_clients) > 0:
				edge_server.model.load_state_dict(new_weights[i])


	def send_cloud_model_to_edge_servers(self):
		for edge_server in self.edge_servers:
			edge_server.model = self.copy_model(self.model).to(device)


	def send_cloud_model_to_clients(self):
		for client in self.clients:
			model = self.copy_model(self.model)
			client.model["model"] = [model.to(device)]


	def send_edge_models_to_clients(self):
		for edge_server in self.edge_servers:
			for client_id in edge_server.connected_clients:
				
				model = self.copy_model(edge_server.model).to(device)
				
				if self.clients[client_id].model["model"] == None:
					self.clients[client_id].model["model"] = [model]
				else:
					self.clients[client_id].model["model"].append(model)


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
			client = ClientNode(i, self.learning_rate)
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

		edge_assignment_type = None

		if self.edges_exchange != 0:
			print("---- [ASSIGNMENT] Link edge servers ----")
			edge_assignment = None

			if self.edges_exchange == 1:
				edge_assignment_type = "erdos"
				edge_assignment = EdgeServerAssignment().random_edge_assignment_erdos_renyi(self.edge_servers, p=self.edges_param)
			elif self.edges_exchange == 2:
				edge_assignment_type = "barabasi"
				edge_assignment = EdgeServerAssignment().random_edge_assignment_barabasi_albert(self.edge_servers, m=self.edges_param)
			elif self.edges_exchange == 3:
				edge_assignment_type = "d-regular"
				edge_assignment = EdgeServerAssignment().random_edge_assignment_degree_k(self.edge_servers, k=self.edges_param)
			
			print(edge_assignment)
			alpha = self.calculate_constant_edge_weights(edge_assignment)
		
		# Send model to clients
		print("---- [DELIVER MODEL] Send global model to clients ----")
		self.send_cloud_model_to_clients()

		# Assigning clients to edge server
		print("---- [ASSIGNMENT] Link clients to edge servers ----")
		client_assignment = ClientAssignment().load("client_assignment.npy", self.edge_servers)
		# client_assignment = ClientAssignment().random_clients_servers_assign(self.clients, self.edge_servers)
		# client_assignment = ClientAssignment().shortest_distance_clients_servers_assign(self.clients, self.edge_servers)
		# client_assignment = ClientAssignment().multiple_edges_assignment(self.clients, self.edge_servers, k=3, alpha=0.0, no_local_epochs=5)
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

			# Train each worker
			for i, client in tqdm(enumerate(self.clients)):
				client.train()
		
			# Average models at edge servers
			if (epoch+1) % self.edge_update == 0:
				print("---- [UPDATE MODEL] Send local models to edge servers ----")
				for edge_server in self.edge_servers:
					models = [self.clients[client_id].model["model"] for client_id in edge_server.connected_clients]
					edge_server.model = self.average_models(models)

				# Edge servers exchange weights
				if self.edges_exchange:
					print("---- [UPDATE MODEL] Edge servers exchange weights ----")
					self.distributed_edges_average(alpha)

				for client in self.clients:
					client.clear_model()

				if (epoch+1)% self.global_update != 0:
					# Send the edge servers' models to all the workers	
					print("---- [DELIVER MODEL] Send edge server models to clients ----")
					self.send_edge_models_to_clients()

			# Average models at cloud servers
			if (epoch+1) % self.global_update == 0:
				print("---- [UPDATE MODEL] Send edge servers models to cloud server ----")
				models = [edge_server.model for edge_server in self.edge_servers if len(edge_server.connected_clients) > 0]
				self.model = self.average_models(models)

				# Validate new model
				accuracy = self.validate(load_weight=False)
				accuracy_logs.append(accuracy)
				if accuracy > best_acc:
					best_acc = accuracy
					self.save_model()

				np.save(edge_assignment_type + "accuracy_logs.npy", accuracy_logs)		
				
				# Clear models
				for client in self.clients:
					client.clear_model()

				for edge_server in self.edge_servers:
					edge_server.clear_model()

				# Send the global model to edge servers
				print("---- [DELIVER MODEL] Send global model to clients ----")
				self.send_cloud_model_to_clients()

		loss_dir = edge_assignment_type + "_train_loss"
		if not os.path.exists(loss_dir):
			os.makedirs(loss_dir)

		for client in self.clients:
			client.save_training_loss(loss_dir) 
		
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
				images = images.view(images.shape[0], -1)
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