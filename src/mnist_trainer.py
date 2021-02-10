import os

import torch
import torch.nn.functional as F
from torch import nn, optim

from mnist_model import Model
from mnist_data_loader import MNIST_DataLoader

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer:
	def __init__(self, batch_size, lr, num_epochs, model_weight_dir):
		self.model = Model().to(device)
		self.data_loader = MNIST_DataLoader(batch_size, workers=None)
		
		self.lr = lr
		self.batch_size = batch_size
		self.num_epochs = num_epochs
		self.model_weight_dir = model_weight_dir


	def train(self):
		print("Train in Normal Mode")

		train_data = self.data_loader.prepare_data(train=True)
		
		optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
		criterion = nn.CrossEntropyLoss()

		print("Start training...")

		for epoch in range(self.num_epochs):
			print("Epoch {}/{}".format(epoch+1, self.num_epochs))

			running_loss = 0
			for batch_idx, (images, labels) in enumerate(train_data):
				images, labels = images.to(device), labels.to(device)
				if (batch_idx+1) % 100 == 0: 
					print("Processed {}/{} batches".format(batch_idx+1, len(train_data)))

				optimizer.zero_grad()

				output = self.model(images)
				loss = criterion(output, labels)
				loss.backward()
				optimizer.step()

				running_loss += loss.item()

		print("Saving model...")
		if not os.path.exists(self.model_weight_dir):
			os.makedirs(self.model_weight_dir)
		torch.save(self.model.state_dict(), self.model_weight_dir + f"/weight.pth")
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

		print("Number of corrects: {}/{}".format(corrects, len(test_data)*self.batch_size))
		print("Accuracy: {}%".format(100*corrects/(len(test_data)*self.batch_size)))
		print("-----------------------------------------")