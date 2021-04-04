import numpy as np
import random
import networkx as nx

SEED = 2

class EdgeServerAssignment:

	def assign_neighbor_servers(self, edge_servers, assignment):
		for i in range(len(edge_servers)-1):
			for j in range(i+1, len(edge_servers)):
				if assignment[i][j] == 1:
					edge_servers[i].neighbor_servers.append(j)
					edge_servers[j].neighbor_servers.append(i)

	
	def random_edge_assignment_erdos_renyi(self, edge_servers, p=0.5):
		G = nx.erdos_renyi_graph(len(edge_servers), p=p, seed=SEED)
		assignment = nx.to_numpy_array(G)

		self.assign_neighbor_servers(edge_servers, assignment)

		return assignment


	def random_edge_assignment_barabasi_albert(self, edge_servers, m=5):
		G = nx.barabasi_albert_graph(len(edge_servers), m=m, seed=SEED)
		assignment = nx.to_numpy_array(G)

		self.assign_neighbor_servers(edge_servers, assignment)

		return assignment


	def random_edge_assignment_degree_k(self, edge_servers, k):
		G = nx.random_regular_graph(k, len(edge_servers), seed=SEED)
		assignment = nx.to_numpy_array(G)

		self.assign_neighbor_servers(edge_servers, assignment)
		
		return assignment


	def complete_graph_edge_assignment(self, edge_servers):
		G = nx.complete_graph(len(edge_servers))
		assignment = nx.to_numpy_array(G)

		self.assign_neighbor_servers(edge_servers, assignment)

		return assignment
