import random
import numpy as np
import torch
from tqdm import tqdm

random.seed(1)

class ClientAssignment:

	def load(self, path, edge_servers):
		assignment = np.load(path)
		assignment = np.transpose(assignment)

		for server_id in range(assignment.shape[0]):
			for client_id in range(assignment.shape[1]):
				if assignment[server_id][client_id] == 1:
					edge_servers[server_id].add_client(client_id)

		assignment = np.transpose(assignment)
		return assignment


	def random_clients_servers_assign(self, clients, edge_servers):
		assignment = np.zeros((len(clients), len(edge_servers)), dtype=np.int8)

		for client_id in range(len(clients)):
			random_server_id = random.randint(0, len(edge_servers)-1)
			assignment[client_id][random_server_id] = 1
			edge_servers[random_server_id].add_client(client_id)

		return assignment


	def calculate_distance_matrix(self, clients, edge_servers):	
		edge_server_locations = [edge_server.location for edge_server in edge_servers]

		distance_matrix = []
		
		for client in clients:
			distances = [np.linalg.norm(client.location - edge_server_location) for edge_server_location in edge_server_locations]
			distance_matrix.append(distances)
		
		return np.array(distance_matrix)
	

	def shortest_distance_clients_servers_assign(self, clients, edge_servers, clients_per_server):
		distance_matrix = self.calculate_distance_matrix(clients, edge_servers)
		distance_matrix = np.transpose(distance_matrix)

		assignment = np.zeros((len(clients), len(edge_servers)), dtype=np.int8)
		
		for server_id in range(len(edge_servers)):
			while assignment.sum(axis=0)[server_id] < clients_per_server:
				nearest_client_id = np.argmin(distance_matrix[server_id])
				if np.sum(assignment[nearest_client_id]) == 0:
					assignment[nearest_client_id][server_id] = 1
					edge_servers[server_id].add_client(nearest_client_id)
		
				distance_matrix[server_id][nearest_client_id] = 100

		return assignment


	def k_nearest_edge_servers_assignment(self, clients, edge_servers, k):
		distance_matrix = self.calculate_distance_matrix(clients, edge_servers)

		assignment = np.zeros((len(clients), len(edge_servers)))

		for client_id in range(len(clients)):
			server_indices = np.argpartition(distance_matrix[client_id], k)
			for server_id in server_indices:
				assignment[client_id][server_id] = 1
				edge_servers[server_id].add_client(client_id)

		return assignment


	def k_nearest_edge_servers_assignment_fixed_size(self, clients, edge_servers, k):
		assignment = self.shortest_distance_clients_servers_assign(clients, edge_servers)

		distance_matrix = self.calculate_distance_matrix(clients, edge_servers)

		for client_id in range(len(clients)):
			server_indices = np.argpartition(distance_matrix[client_id], k)
			for server_id in server_indices:
				if assignment[client_id][server_id] == 0:
					assignment[client_id][server_id] = 1
					edge_servers[server_id].add_client(client_id)

				if np.sum(assignment[client_id]) == k:
					break

		return assignment

	
	def random_multiple_edges_assignment(self, clients, edge_servers, k):
		assignment = np.zeros((len(clients), len(edge_servers)))
		for client_id in range(len(clients)):
			random_servers = random.sample(range(len(edge_servers)), k)
			for server_id in random_servers:
				edge_servers[server_id].add_client(client_id)
				assignment[client_id][server_id] = 1

		return assignment


	def weight_difference(self, model_A, model_B):
		model_A_vector, model_B_vector = [], []
		with torch.no_grad():
			for param_A, param_B in zip(model_A.parameters(), model_B.parameters()):
				model_A_vector.append(torch.flatten(param_A.data))
				model_B_vector.append(torch.flatten(param_B.data))

			model_A_vector = torch.cat(model_A_vector)
			model_B_vector = torch.cat(model_B_vector)

			difference = torch.norm(model_A_vector - model_B_vector)

		return difference


	def calculate_weight_difference_matrix(self, clients):
		difference_matrix = np.zeros((len(clients), len(clients)))

		for i in range(len(clients)):
			client_A = clients[i]
			model_A = client_A.model["model"]
			
			for j in range(i+1, len(clients)):
				
				client_B = clients[j]
				model_B = client_B.model["model"]

				difference = self.weight_difference(model_A, model_B)
				difference_matrix[i][j] = difference
				difference_matrix[j][i] = difference

		return difference_matrix


	def multiple_edges_assignment(self, clients, edge_servers, k, alpha, no_local_epochs):
		print("---- Assignment Phase Model Training ----")

		# Train the local models for a few epochs
		for i, client in tqdm(enumerate(clients)):
			client.train(no_local_epochs)

		# Calculate the distances between workers and edge servers
		print("-- Calculate distance matrix")
		distance_matrix = self.calculate_distance_matrix(clients, edge_servers)

		# Calculate the weight differences between workers
		print("-- Calculate weight difference matrix")
		weight_difference_matrix = self.calculate_weight_difference_matrix(clients)
		np.save("weight_diff.npy", weight_difference_matrix)

		# Reset the assignment
		for edge_server in edge_servers:
			edge_server.connected_clients = []

		assignment = np.zeros((len(clients), len(edge_servers)))

		# Start the assignment
		print("-- Assign workers to edge server")

		for client_id in range(len(clients)):
			cost = alpha*distance_matrix[client_id][:] + (1-alpha)*np.sum([assignment[client_id][s]*(1-assignment[j][s])*weight_difference_matrix[client_id][j] for j in range(client_id) for s in range(len(edge_servers))])
			server_indices = np.argpartition(cost, k)
			for server_id in server_indices[:k]:
				assignment[client_id][server_id] = 1
				edge_servers[server_id].add_client(client_id)

		# Clear models
		for client in clients:
			client.model["model"] = None

		return assignment
