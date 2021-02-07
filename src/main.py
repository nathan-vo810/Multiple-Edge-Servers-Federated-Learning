import argparse

from mnist_federated_trainer import FederatedTrainer
from mnist_trainer import Trainer

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 5
MODEL_WEIGHT_PATH = "./weight.pth"

NUM_WORKERS = 3

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', dest='mode', help='Mode: normal/federated-iid/federated-non-iid', default='normal')
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
		trainer = Trainer(BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, MODEL_WEIGHT_PATH)
	elif args.mode == 'federated-iid':
		trainer = FederatedTrainer(BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, MODEL_WEIGHT_PATH, NUM_WORKERS, iid=True)
	else:
		trainer = FederatedTrainer(BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, MODEL_WEIGHT_PATH, NUM_WORKERS, iid=False)

	if t_or_f(args.train):		
		trainer.train()

	trainer.validate()

if __name__ == '__main__':
	main(parse_args())

