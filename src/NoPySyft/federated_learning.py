import os
import torch
import copy
from tqdm import tqdm

from torch import nn

from mnist_model import CNNModel
from client_node import ClientNode
from data_loader import MNISTDataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

class Trainer:
	def __init__(self, no_clients, learning_rate, batch_size, epochs, no_local_epochs, model_weight_dir):
		self.model = CNNModel().to(device)
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.epochs = epochs
		self.data_loader = MNISTDataLoader(batch_size)
		self.clients = self.generate_clients(no_clients)
		self.no_local_epochs = no_local_epochs
		self.model_weight_dir = model_weight_dir


	def generate_clients(self, no_clients):
		clients = [] 
		for i in range(no_clients):
			client = ClientNode(self.learning_rate)
			clients.append(client)

		return clients


	def send_model_to_clients(self):
		for client in self.clients:
			client.model["model"] = copy.deepcopy(self.model)


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


	def train(self):

		# Load and distribute data to clients
		train_data = self.data_loader.prepare_federated_pathological_non_iid(len(self.clients))
		train_data = self.data_loader.prepare_iid_data(len(self.clients))

		print("Distributing data...")
		for client_id, client_data in tqdm(train_data.items()):
			self.clients[client_id].data = client_data

		accuracy_logs = []
		best_acc = 0

		print("Start training...")
		for epoch in range(self.epochs):
			print(f"Epoch {epoch+1}/{self.epochs}")
			# Send model to all clients
			self.send_model_to_clients()

			# Update local model for several epochs
			print("Local updating on clients")
			for i in tqdm(range(len(self.clients))):
				for epoch in range(self.no_local_epochs):	
					self.clients[i].train(device)

			# Get back and average the local models
			client_models = [client.model["model"] for client in self.clients]
			self.model = self.average_models(client_models)

			# Validate new model
			accuracy = self.validate(load_weight=False)
			accuracy_logs.append(accuracy)
			if accuracy > best_acc:
				best_acc = accuracy
				self.save_model()


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