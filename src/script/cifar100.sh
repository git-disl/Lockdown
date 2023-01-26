#!/bin/bash
cd  ../                            # Change to working directory
nohup python federated.py   --method fedavg --aggr avg   --data cifar100
nohup python federated.py   --method lockdown --aggr mask_avg   --data cifar100
nohup python federated.py   --method krum --aggr avg   --data cifar100
nohup python federated.py   --method rlr --aggr avg  --theta 8 --data cifar100
wait