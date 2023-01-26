import copy
import math

import torch
import utils
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np


class Agent():
    def __init__(self, id, args, train_dataset=None, data_idxs=None, mask=None):
        self.id = id
        self.args = args
        self.error = 0
        # get datasets, fedemnist is handled differently as it doesn't come with pytorch
        if train_dataset is None:
            self.train_dataset = torch.load(f'../data/Fed_EMNIST/user_trainsets/user_{id}_trainset.pt')

            # for backdoor attack, agent poisons his local dataset
            if self.id < args.num_corrupt:
                utils.poison_dataset(self.train_dataset, args, data_idxs, agent_idx=self.id)

        else:
            if self.args.data != "tinyimagenet":
                self.train_dataset = utils.DatasetSplit(train_dataset, data_idxs)
                # for backdoor attack, agent poisons his local dataset
                if self.id < args.num_corrupt:
                    utils.poison_dataset(train_dataset, args, data_idxs, agent_idx=self.id)
            else:
                self.train_dataset = utils.DatasetSplit(train_dataset, data_idxs, runtime_poison=True, args=args,
                                                        client_id=id)

        # get dataloader
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True, \
                                       num_workers=args.num_workers, pin_memory=False, drop_last=True)
        # size of local dataset
        self.n_data = len(self.train_dataset)

        self.mask = copy.deepcopy(mask)

    def screen_gradients(self, model, temparature=10, alpha=0.1):
        model.eval()
        # # # train and update
        criterion = nn.CrossEntropyLoss()
        gradient = {name: 0 for name, param in model.named_parameters()}
        # # sample 10 batch  of data
        batch_num = 0
        for _, (x, labels) in enumerate(self.train_loader):
            batch_num+=1
            model.zero_grad()
            x, labels = x.to(self.args.device), labels.to(self.args.device)
            log_probs = model.forward(x)
            minibatch_loss = criterion(log_probs, labels.long())
            loss = minibatch_loss
            loss.backward()
            for name, param in model.named_parameters():
                gradient[name] += param.grad.data
        return gradient

    def update_mask(self, masks, num_remove, gradient=None):
        new_mask = copy.deepcopy(masks)
        for name in masks:
            if num_remove[name] == 0:
                continue
            if self.args.dis_check_gradient:
                temp = torch.where(masks[name] == 0, torch.ones_like(masks[name]), torch.zeros_like(masks[name]))
                idx = torch.multinomial(temp.flatten().to(self.args.device), num_remove[name], replacement=False)
                new_mask[name].view(-1)[idx] = 1
            else:
                temp = torch.where(masks[name].to(self.args.device) == 0, torch.abs(gradient[name]),
                                   -100000 * torch.ones_like(gradient[name]))
                sort_temp, idx = torch.sort(temp.view(-1), descending=True)
                new_mask[name].view(-1)[idx[:num_remove[name]]] = 1
        return new_mask

    def fire_mask(self, weights, masks, round):
        new_mask = copy.deepcopy(masks)
        if round<=self.args.rounds/2:
            drop_ratio = self.args.anneal_factor / 2 * (1 + np.cos((round * np.pi) / (self.args.rounds/2)))
        else:
            drop_ratio = 0
        # logging.info(drop_ratio)
        num_remove = {}
        for name in masks:
            num_non_zeros = torch.sum(masks[name])
            num_remove[name] = math.ceil(drop_ratio * num_non_zeros)
            if num_remove[name]>0:
                temp_weights = torch.where(masks[name] > 0, torch.abs(weights[name]),
                                           100000 * torch.ones_like(weights[name]))
                x, idx = torch.sort(temp_weights.view(-1).to(self.args.device))
                new_mask[name].view(-1)[idx[:num_remove[name]]] = 0
                weights[name]*=new_mask[name]
        return new_mask, num_remove

    def local_train(self, global_model, criterion, round=None, temparature=10, alpha=0.3):
        for name, param in global_model.named_parameters():
            self.mask[name] =self.mask[name].to(self.args.device)
            param.data = param.data * self.mask[name]
        global_model.train()
        initial_global_model_params = parameters_to_vector([ global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        lr = self.args.client_lr* (self.args.lr_decay)**round
        optimizer = torch.optim.SGD(global_model.parameters(), lr=lr, weight_decay=self.args.wd)
        # pure_gradient_update = torch.zeros_like(initial_global_model_params)
        for _ in range(self.args.local_ep):
            for _, (inputs, labels) in enumerate(self.train_loader):
                optimizer.zero_grad()
                inputs, labels = inputs.to(device=self.args.device), \
                                 labels.to(device=self.args.device)
                outputs = global_model(inputs)
                minibatch_loss = criterion(outputs, labels)
                if round<=self.args.rounds/2:
                    for name, param in global_model.named_parameters():
                        minibatch_loss += self.args.se_threshold * torch.norm(param, 1)
                loss = minibatch_loss
                loss.backward()
                for name, param in global_model.named_parameters():
                    param.grad.data = self.mask[name] * param.grad.data
                optimizer.step()
        with torch.no_grad():
            after_train = parameters_to_vector([ global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
            self.update = (after_train - initial_global_model_params)
        self.mask, num_remove = self.fire_mask(global_model.state_dict(), self.mask, round)
        gradient = self.screen_gradients(global_model,temparature=temparature, alpha=alpha)
        self.mask = self.update_mask(self.mask, num_remove, gradient)
        for name in self.mask:
            self.mask[name] = self.mask[name].to("cpu")
        return self.update

