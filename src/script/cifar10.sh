#!/bin/bash
cd  ../                            # Change to working directory
nohup python federated.py   --method fedavg --aggr avg   --data cifar10 &
nohup python federated.py   --method lockdown --aggr mask_avg   --data cifar10 &
nohup python federated.py   --method fedavg --aggr krum   --data cifar10 &
nohup python federated.py   --method rlr --aggr avg  --theta 8 --data cifar10 &
wait