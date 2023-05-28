#!/bin/bash
cd  ../                            # Change to working directory
nohup python federated.py   --method fedavg --aggr avg   --data fmnist &
nohup python federated.py   --method lockdown --aggr avg   --data fmnist &
nohup python federated.py   --method fedavg --aggr krum   --data fmnist &
nohup python federated.py   --method rlr --aggr avg  --theta 8 --data fmnist &
wait