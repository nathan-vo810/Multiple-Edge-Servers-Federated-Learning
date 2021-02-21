# Multiple Edge Servers Assignment for Local Device in Hierarchical Federated Learning

This repository contains the source code for Multiple Edge Servers Assignment for Local Device in Hierarchical Federated Learning

## Getting Started
Install PySyft to support Federated Learning. Follow the tutorial [here](https://pysyft.readthedocs.io/en/latest/installing.html)

## Running The Code
To run the code, go to terminal and type 
-- For training: ```python3 main.py --mode=MODE --train``` 
-- For validating: ```python3 main.py --mode=MODE --no-train```

There are currently 5 modes:
1. normal
-- In this mode, the trainer trains on a single machine.
2. federated-iid
-- This mode distributes the data set to the workers in the iid fashion. The trainer trains the model on the workers' machine and the model is sent to the central server for averaging.
3. federated-non-iid	
-- This mode distributes the data set to the workers in the non-iid fashion. The trainer trains the model on the workers' machine and the model is sent to the central server for averaging.
4. hierarchical-iid
-- This mode is similar to the federated-iid mode but with an additional layer of edge servers between the central server and the workers. The edge server will average the workers' models after every I epochs, and the central server will average the edge servers' models after every G epochs.
5. hierarchical-non-iid
-- This mode is similar to the federated-non-iid mode but with an additional layer of edge servers between the central server and the workers. The edge server will average the workers' models after every I epochs, and the central server will average the edge servers' models after every G epochs.
