import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset

from mnist_model import Model


class Trainer:
	def __init__(self, batch_size, lr, num_epochs, model_weight_path):
		self.model = Model()
		
		self.lr = lr
		self.batch_size = batch_size
		self.num_epochs = num_epochs
		self.model_weight_path = model_weight_path


	def prepare_data(self, train=True):
		transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
		data = datasets.MNIST('./data', train=train, download=True, transform=transform)

		data_loader = DataLoader(data, batch_size=self.batch_size, shuffle=True)

		return data_loader


	def train(self):
		print("Train in Normal Mode")

		train_data = self.prepare_data(train=True)
		
		optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
		criterion = nn.CrossEntropyLoss()

		print("Start training...")

		for epoch in range(self.num_epochs):
			print("Epoch {}/{}".format(epoch+1, self.num_epochs))

			running_loss = 0
			for batch_idx, (images, labels) in enumerate(train_data):
				if batch_idx % 100 == 0: 
					print("Processed {}/{} batches".format(batch_idx, len(train_data)))

				optimizer.zero_grad()

				output = self.model(images)
				loss = criterion(output, labels)
				loss.backward()
				optimizer.step()

				running_loss += loss.item()

		print("Saving model...")
		torch.save(self.model.state_dict(), MODEL_WEIGHT_PATH)
		print("Finish training!")

	def validate(self):
		print("Start validating...")
		self.model.load_state_dict(torch.load(self.model_weight_path))
		self.model.eval()
		corrects = 0

		test_data = self.prepare_data(train=False)

		with torch.no_grad():
			for batch_idx, (images, labels) in enumerate(test_data):
				print("Predicting batch {}/{}".format(batch_idx+1, len(test_data)))
				output = self.model(images)
				pred = output.argmax(dim=1)
				corrects += pred.eq(labels.view_as(pred)).sum().item()

		print("Number of corrects: {}/{}".format(corrects, len(test_data)*self.batch_size))
		print("Accuracy: {}%".format(100*corrects/(len(test_data)*self.batch_size)))