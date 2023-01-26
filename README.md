

# Lockdown: Backdoor Defense for Federated Learning with Isolated Training Subspace
This is the repo for the paper "Lockdown: Backdoor Defense for Federated Learning with Isolated Subspace Training".

## Algorithm overview
The overall procedure can be summarized into four main steps. i) Isolated subspace training. ii)Subspace searching. iii) Aggregation. iv) Model cleaning with consensus fusion.
The following figure illustrates the overall process. 
<div align=center><img width="750" height="500" src="https://github.com/LockdownAuthor/Lockdown/blob/main/materials/system.png"/></div>


## Package requirement
* PyTorch 
* Numpy
* TorchVision

## Data  preparation
Dataset FashionMnist and CIFAR10/100 will be automatically downloaded with TorchVision.

## Command to run
The following code run lockdown in its default setting
```
python federated.py  --method lockdown \
                     --aggr mask_avg \
                      --mask_init ERK \
                       --se_threshold 1e-4 \
                        --theta 25 \
```
You can also find script in directory `src/script`.

## Logging and checkpoint
The logging files will be contained in `src/logs`. Benign accuracy, ASR, and Backdoor accuracy will be tested in every round.
For Lockdown, the three metrics correspond to the following logging format:
```
| Clean Val_Loss/Val_Acc: (Benign loss) / (Benign accuracy) |
| Clean Attack Success Ratio: (ASR loss)/ (ASR) |
| Clean Poison Loss/Clean Poison accuracy:: (Backdoor Loss)/ (Backdoor Acc)|
```
Model checkpoints will be saved every 25 rounds in the directory `src/checkpoint`.

## Acknowledgment
The codebase is modified and adapted from one of our baselines RLR (https://github.com/TinfoilHat0/Defending-Against-Backdoors-with-Robust-Learning-Rate)





