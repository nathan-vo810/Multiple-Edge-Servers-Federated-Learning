## Multiple Edge Servers Assignment for Local Device in Hierarchical Federated Learning

This folder contains the source code for Multiple Edge Servers Assignment for Local Device in Hierarchical Federated Learning

### Requirements
* PyTorch
* NumPy
* tqdm


### Configurations
To change the configurations of the training process, go to file ```main.py```

Available configurations:
1. Number of edge servers: NUM_EDGE_SERVERS (default = 10)
2. Number of clients: NUM_CLIENTS (default = 100)
3. Update edge servers' models after every I epochs: EDGE_UPDATE_AFTER_EVERY (default = 2)
4. Update cloud server's model after every G epochs: GLOBAL_UPDATE_AFTER_EVERY (default = 4)
5. Number of epochs: NUM_EPOCHS (default = 20)
6. Batch size: BATCH_SIZE (default = 10)
7. Learning rate: LEARNING_RATE (default = 1e-3)


### Running The Code
To run the code, go to terminal and type 
* For training: ```python3 main.py --mode=MODE``` 
Currently supporting modes:
    -  federated-non-iid (default)
    -  hierarchical-non-iid