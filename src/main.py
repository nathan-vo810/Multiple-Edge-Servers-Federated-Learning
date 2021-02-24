import argparse

from mnist_federated_trainer import FederatedTrainer
from mnist_federated_hierarchical_trainer import FederatedHierachicalTrainer
from mnist_trainer import Trainer

BATCH_SIZE = 10
LEARNING_RATE = 1e-3
NUM_ROUNDS = 200
NUM_EPOCHS = 20
MODEL_WEIGHT_DIR = "../weight"
NUM_WORKERS = 100

##############################
NUM_EDGE_SERVERS = 10
WORKERS_PER_SERVER = 10
EDGE_UPDATE_AFTER_EVERY = 1
GLOBAL_UPDATE_AFTER_EVERY = 2

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', dest='mode', help='Mode: normal/federated-iid/federated-non-iid/hierarchical-iid/hierarchical-non-iid', default='normal')
	parser.add_argument('--train', dest='train', action='store_true')
	parser.add_argument('--no-train', dest='train', action='store_false')
	parser.set_defaults(feature=True)

	return parser.parse_args()

def t_or_f(arg):
	ua = str(arg).upper()
	if 'TRUE'.startswith(ua):
		return True
	elif 'FALSE'.startswith(ua):
		return False
	else:
		pass

def main(args):
	if args.mode == 'normal':
		trainer = Trainer(BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, MODEL_WEIGHT_DIR)
	elif args.mode == 'federated-iid':
		trainer = FederatedTrainer(BATCH_SIZE, LEARNING_RATE, NUM_ROUNDS, NUM_EPOCHS, MODEL_WEIGHT_DIR, NUM_WORKERS, iid=True, parallel=False)
	elif args.mode == 'federated-non-iid':
		trainer = FederatedTrainer(BATCH_SIZE, LEARNING_RATE, NUM_ROUNDS, NUM_EPOCHS, MODEL_WEIGHT_DIR, NUM_WORKERS, iid=False, parallel=True)
	elif args.mode == 'hierarchical-iid':
		trainer = FederatedHierachicalTrainer(BATCH_SIZE, LEARNING_RATE, NUM_ROUNDS, NUM_EPOCHS, MODEL_WEIGHT_DIR, NUM_WORKERS, NUM_EDGE_SERVERS, WORKERS_PER_SERVER, EDGE_UPDATE_AFTER_EVERY, GLOBAL_UPDATE_AFTER_EVERY, iid=True)
	else:
		trainer = FederatedHierachicalTrainer(BATCH_SIZE, LEARNING_RATE, NUM_ROUNDS, NUM_EPOCHS, MODEL_WEIGHT_DIR, NUM_WORKERS, NUM_EDGE_SERVERS, WORKERS_PER_SERVER, EDGE_UPDATE_AFTER_EVERY, GLOBAL_UPDATE_AFTER_EVERY, iid=False)

	if t_or_f(args.train):		
		trainer.train()

	trainer.validate(load_weight=True)

if __name__ == '__main__':
	main(parse_args())

