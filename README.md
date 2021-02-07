# Multiple Edge Servers Assignment for Local Device in Hierarchical Federated Learning

This repository contains the source code for Multiple Edge Servers Assignment for Local Device in Hierarchical Federated Learning

To run the code, go to terminal and type 
For training: ```python3 main.py --mode=MODE --train``` 
For validating: ```python3 main.py --mode=MODE --no-train```

There are currently 3 modes:
1. normal
-- In this mode, the trainer trains on a single machine.
2. federated-iid
-- This mode distributes the data set to the workers in the iid fashion. The trainer trains the model on the workers' machine and aggregate the model. 
3. federated-non-iid	
-- This mode distributes the data set to the workers in the non-iid fashion. The trainer trains the model on the workers' machine and aggregate the model. 