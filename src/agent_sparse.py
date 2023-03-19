import copy
import math
import time

import torch
import models
import utils
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import logging


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
                    self.clean_backup_dataset = copy.deepcopy(train_dataset)
                    self.data_idxs = data_idxs
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

    def check_poison_timing(self, round):
        if round > self.args.cease_poison:
            self.train_dataset = utils.DatasetSplit(self.clean_backup_dataset, self.data_idxs)
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True, \
                                           num_workers=self.args.num_workers, pin_memory=False, drop_last=True)

    def screen_gradients(self, model, temparature=10, alpha=0.1):
        model.eval()
        # # # train and update
        criterion = nn.CrossEntropyLoss()
        gradient = {name: 0 for name, param in model.named_parameters()}
        # # sample 10 batch  of data
        batch_num = 0
        for _, (x, labels) in enumerate(self.train_loader):
            batch_num += 1
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
            if self.args.dis_check_gradient or gradient == None:
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
        if round <= self.args.rounds / 2:
            drop_ratio = self.args.anneal_factor / 2 * (1 + np.cos((round * np.pi) / (self.args.rounds / 2)))
        else:
            drop_ratio = 0
        # logging.info(drop_ratio)
        num_remove = {}
        for name in masks:
            num_non_zeros = torch.sum(masks[name])
            num_remove[name] = math.ceil(drop_ratio * num_non_zeros)
            if num_remove[name] > 0:
                temp_weights = torch.where(masks[name] > 0, torch.abs(weights[name]),
                                           100000 * torch.ones_like(weights[name]))
                x, idx = torch.sort(temp_weights.view(-1).to(self.args.device))
                new_mask[name].view(-1)[idx[:num_remove[name]]] = 0
                weights[name] *= new_mask[name]
        return new_mask, num_remove

    def local_train(self, global_model, criterion, round=None, temparature=10, alpha=0.3, global_mask=None,
                    neurotoxin_mask=None, updates_dict=None):
        """ Do a local training over the received global model, return the update """
        # if  self.id<self.args.num_corrupt and self.mask_update!= None:
        #     initial_global_model_params_local = self.mask_update.to(self.args.device) + initial_global_model_params.to(self.args.device) * (1-self.previous_mask.to(self.args.device))
        #     vector_to_parameters(initial_global_model_params_local, global_model.parameters())
        if self.id < self.args.num_corrupt:
            self.check_poison_timing(round)
        for name, param in global_model.named_parameters():
            self.mask[name] = self.mask[name].to(self.args.device)
            param.data = param.data * self.mask[name]
        global_model.train()
        initial_global_model_params = parameters_to_vector(
            [global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        # print(torch.sum(initial_global_model_params))
        lr = self.args.client_lr * (self.args.lr_decay) ** round
        # logging.info(lr)
        optimizer = torch.optim.SGD(global_model.parameters(), lr=lr, weight_decay=self.args.wd)
        # pure_gradient_update = torch.zeros_like(initial_global_model_params)

        if self.id < self.args.num_corrupt and (
                self.args.attack == "omniscient" or self.args.attack == "fix_mask" or self.args.attack == "neurotoxin"):
            do_l1_norm = False
        else:
            do_l1_norm = True

        for _ in range(self.args.local_ep):
            # start = time.time()
            for _, (inputs, labels) in enumerate(self.train_loader):

                optimizer.zero_grad()
                inputs, labels = inputs.to(device=self.args.device), \
                                 labels.to(device=self.args.device)

                outputs = global_model(inputs)

                # KL_loss = nn.KLDivLoss()(F.log_softmax(outputs/temparature, dim=1),F.softmax(dense_outputs/temparature,dim=1))
                minibatch_loss = criterion(outputs, labels)

                if round <= self.args.rounds / 2 and do_l1_norm:
                    for name, param in global_model.named_parameters():
                        minibatch_loss += self.args.se_threshold * torch.norm(param, 1)
                loss = minibatch_loss
                loss.backward()
                for name, param in global_model.named_parameters():
                    param.grad.data = self.mask[name] * param.grad.data
                optimizer.step()
            # end = time.time()
            # print(end - start)

        with torch.no_grad():
            after_train = parameters_to_vector(
                [global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
            self.update = (after_train - initial_global_model_params)
            if "scale" in self.args.attack:
                logging.info("scale update for" + self.args.attack.split("_", 1)[1] + " times")
                if self.id < self.args.num_corrupt:
                    self.update = int(self.args.attack.split("_", 1)[1]) * self.update

        if self.id < self.args.num_corrupt:
            if self.args.attack == "fix_mask":
                self.mask = self.mask

            elif self.args.attack == "omniscient":
                if len(global_mask):
                    self.mask = copy.deepcopy(global_mask)
                else:
                    self.mask = self.mask
            elif self.args.attack == "neurotoxin":
                if len(neurotoxin_mask):
                    self.mask = neurotoxin_mask
                else:
                    self.mask = self.mask
            elif self.args.attack == "snooper":
                if len(updates_dict):
                    self.mask, num_remove = self.fire_mask(updates_dict, self.mask, round)
                    self.mask = self.update_mask(self.mask, num_remove)
            else:
                self.mask, num_remove = self.fire_mask(global_model.state_dict(), self.mask, round)
                gradient = self.screen_gradients(global_model, temparature=temparature, alpha=alpha)
                # gradient = {name: torch.zeros_like(param) for name, param in global_model.named_parameters()}
                self.mask = self.update_mask(self.mask, num_remove, gradient)

        else:
            self.mask, num_remove = self.fire_mask(global_model.state_dict(), self.mask, round)
            gradient = self.screen_gradients(global_model, temparature=temparature, alpha=alpha)
            # gradient = {name: torch.zeros_like(param) for name, param in global_model.named_parameters()}
            self.mask = self.update_mask(self.mask, num_remove, gradient)
        for name in self.mask:
            self.mask[name] = self.mask[name].to("cpu")
        return self.update

