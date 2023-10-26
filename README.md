

# Lockdown: Backdoor Defense for Federated Learning with Isolated Subspace Training
This is the repo for the code and datasets used in the paper [Lockdown: Backdoor Defense for Federated Learning with Isolated Subspace Training](https://neurips.cc/virtual/2023/poster/71476), accepted by the NeurIPS 2023.

## Algorithm overview
The overall procedure can be summarized into four main steps. i) Isolated subspace training. ii)Subspace searching. iii) Aggregation. iv) Model cleaning with consensus fusion.
The following figure illustrates the overall process. 
<div align=center><img width="700" height="450" src="https://github.com/git-disl/Lockdown/blob/main/materials/system.png"/></div>

## Get started
### Package requirement
* PyTorch 
* Numpy
* TorchVision

### Data  preparation
Dataset FashionMnist and CIFAR10/100 will be automatically downloaded with TorchVision.

### Command to run
The following code run lockdown in its default setting
```
python federated.py  --method lockdown 
```
You can also find script in directory `src/script`.

### Files organization
* The main simulation program is in `decentralized.py`, where we initialize the benign and poison dataset, call clients to do local training, call aggregator to do aggregation, do consensus fusion before testing, etc.

* The Lockdown's client local training logistic is in `agent_sparse.py`. 

* The vanilla FedAvg' client local training logistic is in `agent.py`. 

* The aggregation logistic is in `aggregation.py`, where we implement multiple defense baselines. 

* The data poisoning, data preparation and data distribution logistic is in `utils.py`.

### Logging and checkpoint
The logging files will be contained in `src/logs`. Benign accuracy, ASR, and Backdoor accuracy will be tested in every round.
For Lockdown, the three metrics correspond to the following logging format:
```
| Clean Val_Loss/Val_Acc: (Benign loss) / (Benign accuracy) |
| Clean Attack Success Ratio: (ASR loss)/ (ASR) |
| Clean Poison Loss/Clean Poison accuracy:: (Backdoor Loss)/ (Backdoor Acc)|
```
Model checkpoints will be saved every 25 rounds in the directory `src/checkpoint`.


## Q&A

If you have any questions, you can either open an issue or contact me (thuang374@gatech.edu), and I will reply as soon as I see the issue or email.

## Acknowledgment
The codebase is modified and adapted from one of our baselines [RLR](https://github.com/TinfoilHat0/Defending-Against-Backdoors-with-Robust-Learning-Rate).

## License
Lockdown is completely free and released under the [MIT License](https://github.com/git-disl/Lockdown/blob/main/materials/license).




