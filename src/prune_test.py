import random

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
from resnet9 import ResNet9_trackbn, ResNet9_tinyimagenet, ResNet9
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib import cm

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
            index = torch.where(channel_lips > channel_lips.mean() + u * channel_lips.std())[0]
            params[name + '.weight'][index] = 0
            sparse_num += torch.numel(params[name + '.weight'][index])
            # print(index)
            total += torch.numel(m.weight)
    print(sparse_num / total)
    print(channel_lips)
    weights_norm = np.round(np.sort(weights_norm), decimals=4).tolist()
    print(weights_norm)
    net.load_state_dict(params)


def WMP(net, k):
    sparse_num = 0
    total = 0
    params = net.state_dict()
    for name in params:
        x = params[name].view(-1)
        # Sort the tensor in ascending order
        sorted_indices = torch.argsort(x)
        prune = int(k * torch.numel(x))
        # Get the indices of the lowest k elements
        pruned_indices = sorted_indices[:prune]
        mask = torch.ones_like(x)
        # Set the values at the pruned indices to 0
        mask[pruned_indices] = 0
        x.data *= mask

    # print(sparse_num/total)
    net.load_state_dict(params)


def WMP(net, k):
    sparse_num = 0
    total = 0
    params = net.state_dict()
    for name in params:
        x = params[name].view(-1)
        # Sort the tensor in ascending order
        sorted_indices = torch.argsort(x)
        prune = int(k * torch.numel(x))
        # Get the indices of the lowest k elements
        pruned_indices = sorted_indices[:prune]
        mask = torch.ones_like(x)
        # Set the values at the pruned indices to 0
        mask[pruned_indices] = 0
        x.data *= mask

    # print(sparse_num/total)
    net.load_state_dict(params)


def warm_up_bn(net, data_loader):
    for _ in range(2):
        # start = time.time()
        for _, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device="cuda:0"), \
                             labels.to(device="cuda:0")
            outputs = net(inputs)


def print_representation(model, partial_poisoned_train_loader, poison_idx):
    # Apply t-SNE to reduce the dimensionality of the features to 2D
    model.eval()
    model = model.cuda()
    # Define your output variable that will hold the output
    dimension = 100

    # Define a hook function. It sets the global out variable equal to the
    # output of the layer to which this hook is attached to.
    def hook(module, input, output):
        global linear_input
        linear_input =  (nn.MaxPool2d(4)(input[0]).view(-1,512) )[:, :dimension]
        return None

    # Your model layer has a register_forward_hook that does the registering for you
    model.classifier.register_forward_hook(hook)
    # print(model.classifier.2)
    # print(model.classifier[2].weight[7,:])
    # plt.hist(model.classifier[2].weight[7,:].detach().cpu().numpy())
    # plt.show()
    # Then you just loop through your dataloader to extract the embeddings
    embeddings = np.zeros(shape=(0, dimension))
    test_predictions = []
    labels = np.zeros(shape=(0))
    gradient  = np.zeros(shape=(512))
    # model.classifier[2].weight.data[:, :63] = torch.zeros_like( model.classifier[2].weight.data[:, :63])
    # model.classifier[2].weight.data[:, 64] = torch.zeros_like(model.classifier[2].weight.data[:, 64])
    for x, y in iter(partial_poisoned_train_loader):
        global linear_input
        x = x.cuda()
        logits = model(x)
        minibatch_loss = torch.nn.CrossEntropyLoss()(logits, y.cuda())
        labels = np.concatenate((labels, y.numpy().ravel()))
        loss = minibatch_loss
        loss.backward()
        embeddings = np.concatenate([embeddings, linear_input.detach().cpu().numpy()], axis=0)
        preds = torch.argmax(logits, dim=1)
        np.concatenate([embeddings, linear_input.detach().cpu().numpy()], axis=0)
        test_predictions.extend(preds.detach().cpu().tolist())
        gradient += model.classifier[2].weight.grad.data[7,:].cpu().numpy()
    # plt.hist(model.classifier[2].weight[7, :].detach().cpu().numpy())
    # plt.show()
    print(embeddings[poison_idx,64])
    # Create a two dimensional t-SNE projection of the embeddings
    test_predictions = np.array(test_predictions)
    poison_idx = np.array(poison_idx)
    tsne = TSNE(2, verbose=1)
    tsne_proj = tsne.fit_transform(embeddings)
    # Plot those points as a scatter plot and label them based on the pred labels
    cmap = cm.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(8, 8))
    num_categories = 10
    for lab in range(num_categories):
        indices = test_predictions == lab
        ax.scatter(tsne_proj[indices, 0], tsne_proj[indices, 1], c=np.array(cmap(lab)).reshape(1, 4), label=lab,
                   alpha=0.5)

    ax.scatter(tsne_proj[poison_idx, 0], tsne_proj[poison_idx, 1], c=np.array(cmap(10)).reshape(1, 4), label=10,
               alpha=0.5)
    ax.legend(fontsize='large', markerscale=2)
    plt.show()



if __name__ == '__main__':

    args = args_parser()
    args.num_agents = 40
    num_target = 10
    args.data = "cifar10"
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)
    logPath = "logs"
    logging.info(args)
    PATH= "checkpoint/Fedavg(p=0.5, N=4, M=40).pt"
    # PATH = "checkpoint/Centralized(p=0.1, N=1, M=1).pt"
    dict = torch.load(PATH)
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

    idxs = (val_dataset.targets >= 0).nonzero().flatten().tolist()
    partial_poisoned_val_set = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs[:1000])
    poison_idx = utils.poison_dataset(partial_poisoned_val_set.dataset, args, idxs[:1000])
    partial_poisoned_train_loader = DataLoader(partial_poisoned_val_set, batch_size=args.bs, shuffle=False,
                                     num_workers=args.num_workers,
                                     pin_memory=False)

    global_model = ResNet9(3, 10).to(args.device)
    state_dict = dict['model_state_dict']
    global_model.load_state_dict(state_dict, strict=False)
    print_representation(global_model,partial_poisoned_train_loader, poison_idx)


def prune_test():
    percents = [1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3]
    ASR = []
    accuracy = []
    # percents= [0.01,0.02,0.03 ]
    for percent_to_select in percents:
        logging.info("percentage  {}".format(percent_to_select))
        global_model = ResNet9(3, 10).to(args.device)
        state_dict = dict['model_state_dict']
        # central_masks = dict['neurotoxin_mask']
        # side_masks = {name: 1-central_masks[name] for name in central_masks}

        global_model.load_state_dict(state_dict, strict=False)
        # warm_up_bn(global_model, train_loader)
        CLP(global_model, percent_to_select)
        # WMP(global_model, percent_to_select)
        criterion = nn.CrossEntropyLoss().to(args.device)

        val_loss, (val_acc, val_per_class_acc), _ = utils.get_loss_n_accuracy(global_model, criterion, val_loader,
                                                                              args, 0)
        accuracy += [val_acc]
        logging.info(f'| val_acc: {val_loss:.3f} / {val_acc:.3f} |')

        poison_loss, (poison_acc, _), _ = utils.get_loss_n_accuracy(global_model, criterion,
                                                                    poisoned_val_loader,
                                                                    args, 0)
        ASR += [poison_acc]
        cum_poison_acc_mean += poison_acc
        logging.info(f'|  Poison Loss/Poison Acc: {poison_loss:.3f} / {poison_acc:.3f} |')
        # ask the guys to finetune the classifier
        # logging.info(mask_aggrement)
        logging.info('Training has finished!')
    logging.info(ASR)
    logging.info(accuracy)