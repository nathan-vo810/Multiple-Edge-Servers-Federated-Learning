import argparse

from cloud_server import CloudServer
from federated_learning import Trainer

MODEL_WEIGHT_DIR = "../weight"

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', dest='mode', help='Mode: federated-non-iid (default)/hierarchical-non-iid', default='federated-non-iid')
	
	parser.add_argument('--num-edge-servers', dest='num_edges', help='Number of edge servers (default 10)', type=int, default=10)
	parser.add_argument('--num-clients', dest='num_clients', help='Number of client nodes (default 100)', type=int, default=100)
	parser.add_argument('--clients-per-server', dest='clients_per_server', help='Number of clients per server', type=int, default=10)
	
	parser.add_argument('--edge-update', dest='edge_update', help='Number of epoch to update edge servers models (default 2)', type=int, default=2)
	parser.add_argument('--global-update', dest='global_update', help='Number of epoch to update cloud model (default 4)', type=int, default=4)
	
	parser.add_argument('--epochs', dest='epochs', help='Number of epochs (default 400)', type=int, default=400)
	parser.add_argument('--batchsize', dest='batchsize', help='Batch size (default 10)', type=int, default=10)
	parser.add_argument('--lr', dest='lr', help='Learning rate (default 1e-3)', default=1e-3)

	parser.add_argument('--edges-exchange', dest='edges_exchange', help='Enable weight sharing between edge servers', type=int, default=1)
	parser.add_argument('--edges-param', dest='edges_param', help='Parameter for edge servers graph', type=float)
	return parser.parse_args()

def main(args):
	if args.mode == 'federated-non-iid':
		trainer = Trainer(args.num_clients, args.lr, args.batchsize, args.epochs, args.global_update, MODEL_WEIGHT_DIR)
		trainer.train()
	else:
		trainer = CloudServer(args.num_edges, args.num_clients, args.clients_per_server, args.epochs, args.batchsize, args.lr, args.edge_update, args.global_update, args.edges_exchange, args.edges_param, MODEL_WEIGHT_DIR)
		trainer.train()

if __name__ == '__main__':
	main(parse_args())
