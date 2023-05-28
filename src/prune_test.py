import torch
import utils
import models
import math
import copy
import numpy as np
from agent import Agent
from agent_sparse import Agent as Agent_s
from tqdm import tqdm
from options import args_parser
from aggregation import Aggregation
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn as nn
from time import ctime
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from utils import H5Dataset
import sys
from resnet9 import ResNet9_trackbn, ResNet9_tinyimagenet,ResNet9
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
import logging

def CLP(net, u):
    sparse_num = 0
    total = 0
    params = net.state_dict()
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            channel_lips = []
            weights_norm = []
            for idx in range(m.weight.shape[0]):
                weight = m.weight[idx]
                weight = weight.reshape(weight.shape[0], -1).cpu()
                channel_lips.append(torch.svd(weight)[1].max())
                weights_norm.append(float(torch.norm(weight)))
            channel_lips = torch.Tensor(channel_lips)
            
            index = torch.where(channel_lips>channel_lips.mean() + u*channel_lips.std())[0]
            params[name+'.weight'][index] = 0
            sparse_num += torch.numel(params[name+'.weight'][index] )
            # print(index)
            total += torch.numel(m.weight)
    
    print(sparse_num/total)
    print(channel_lips)
    weights_norm =np.round(np.sort(weights_norm), decimals=4).tolist()
    print(weights_norm)
    net.load_state_dict(params)


    
if __name__ == '__main__':
    
    args = args_parser()
    args.num_agents=40
    num_target=10
    args.data="cifar10"
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)
    logPath ="logs"
    logging.info(args)
    # PATH= "checkpoint/AckRatio4_1_Methodfedavg_datacifar10_alpha0.1_Rnd200_Epoch2_inject0.5_dense0.5_Aggavg_se_threshold0.0001_noniidFalse_maskthreshold20_attackr_neurotoxin.pt"
    PATH= "checkpoint/AckRatio4_40_Methodfedavg_datacifar10_alpha0.5_Rnd200_Epoch2_inject0.5_dense0.25_Aggavg_se_threshold0.0001_noniidTrue_maskthreshold20_attackbadnet.pt"
    dict= torch.load(PATH)

    cum_poison_acc_mean = 0

    # load dataset and user groups (i.e., user to data mapping)
    train_dataset, val_dataset = utils.get_datasets(args.data)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers,
                            pin_memory=False)
    # fedemnist is handled differently as it doesn't come with pytorch
    if args.data != 'fedemnist':
        if args.non_iid:
            user_groups = utils.distribute_data_dirichlet(train_dataset, args)
        else:
            user_groups = utils.distribute_data(train_dataset, args, n_classes=num_target)

    # poison the validation dataset
    idxs = (val_dataset.targets != args.target_class).nonzero().flatten().tolist()
    poisoned_val_set = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
    poison_idx = utils.poison_dataset(poisoned_val_set.dataset, args, idxs, poison_all=True)
    poisoned_val_loader = DataLoader(poisoned_val_set, batch_size=args.bs, shuffle=False, num_workers=args.num_workers,
                                     pin_memory=False)
    train_loader = DataLoader(poisoned_val_set, batch_size=args.bs, shuffle=False, num_workers=args.num_workers,
                                     pin_memory=False)
    percents= [ 1, 1.2, 1.4, 1.6, 1.8,  2, 2.2, 2.4,2.6, 2.8, 3 ]
    ASR= []
    accuracy = []
    # percents= [0.01,0.02,0.03 ]
    for percent_to_select in percents:
        logging.info("percentage  {}".format( percent_to_select))
        global_model = ResNet9(3,10).to(args.device)
        state_dict = dict['model_state_dict']
        global_model.load_state_dict(state_dict, strict=False)
        CLP(global_model, percent_to_select)
        criterion = nn.CrossEntropyLoss().to(args.device)
        val_loss, (val_acc, val_per_class_acc), _ = utils.get_loss_n_accuracy(global_model, criterion, val_loader,
                                                                                args, 0)
        accuracy += [ val_acc] 
        logging.info(f'| val_acc: {val_loss:.3f} / {val_acc:.3f} |')
        
        
        poison_loss, (poison_acc, _), _ = utils.get_loss_n_accuracy(global_model, criterion,
                                                                    poisoned_val_loader,
                                                                    args, 0)
        ASR += [ poison_acc] 
        cum_poison_acc_mean += poison_acc
        logging.info(f'|  Poison Loss/Poison Acc: {poison_loss:.3f} / {poison_acc:.3f} |')
        logging.info('Training has finished!')
    logging.info(ASR)
    logging.info(accuracy)






