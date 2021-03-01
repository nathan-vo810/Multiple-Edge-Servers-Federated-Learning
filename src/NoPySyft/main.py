from cloud_server import CloudServer

NUM_EDGE_SERVERS = 10
NUM_CLIENTS = 100

NUM_EPOCHS = 20
BATCH_SIZE = 10
LEARNING_RATE = 1e-3

def main():
	trainer = CloudServer(NUM_EDGE_SERVERS, NUM_CLIENTS, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE)
	trainer.train()

if __name__ == '__main__':
	main()
