# Multiple Edge Servers Assignment for Local Device in Hierarchical Federated Learning

This repository contains the source code for Multiple Edge Servers Assignment for Local Device in Hierarchical Federated Learning

## Getting Started
### Install Anaconda
* Download Anaconda: ```curl -O https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh```
* Activate Anaconda: ```source ~/.bashrc```
* Create the environment: ```conda create -n pysyft python=3.8```
* Activate the environment: ```conda activate pysyft```

### Install PySyft to support Federated Learning
* Install PySyft version 0.2.9: ```pip install "syft<0.3.0"```

## Running The Code
To run the code, go to terminal and type 
* For training: ```python3 main.py --mode=MODE --train``` 
* For validating: ```python3 main.py --mode=MODE --no-train```

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