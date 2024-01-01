import copy
import math
import torch
import utils
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import logging

class Agent():
    def __init__(self, id, args, train_dataset=None, data_idxs=None, mask=None):
        self.id = id
        self.args = args
        self.error = 0
        # poisoned datasets, tinyimagenet is handled differently as the dataset is not loaded into memory
        if self.args.data != "tinyimagenet":
            self.train_dataset = utils.DatasetSplit(train_dataset, data_idxs)
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
        self.num_remove= None

    def check_poison_timing(self, round):
        if round> self.args.cease_poison:
            self.train_dataset = utils.DatasetSplit(self.clean_backup_dataset, self.data_idxs)
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True, \
                                       num_workers=self.args.num_workers, pin_memory=False, drop_last=True)




    def screen_gradients(self, model):
        model.train()
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
        for name in gradient:
            if self.args.dis_check_gradient:
                temp = torch.where(masks[name] == 0, torch.ones_like(masks[name]), torch.zeros_like(masks[name]))
                idx = torch.multinomial(temp.flatten().to(self.args.device), num_remove[name], replacement=False)
                masks[name].view(-1)[idx] = 1
            else:
                temp = torch.where(masks[name].to(self.args.device) == 0, torch.abs(gradient[name]),
                                    -100000 * torch.ones_like(gradient[name]))
                sort_temp, idx = torch.sort(temp.view(-1), descending=True)
                masks[name].view(-1)[idx[:num_remove[name]]] = 1
        return masks
    
    # def init_mask(self,  gradient=None):
    #     for name in self.mask:
    #         num_init = torch.count_nonzero(self.mask[name])
    #         self.mask[name] = torch.zeros_like(self.mask[name])
    #         sort_temp, idx = torch.sort(torch.abs(gradient[name]).view(-1), descending=True)
    #         self.mask[name].view(-1)[idx[:num_init]] = 1
             

    def fire_mask(self, weights, masks, round):
        
        drop_ratio = self.args.anneal_factor / 2 * (1 + np.cos((round * np.pi) / (self.args.rounds)))
    
        # logging.info(drop_ratio)
        num_remove = {}
        for name in masks:
                num_non_zeros = torch.sum(masks[name].to(self.args.device))
                num_remove[name] = math.ceil(drop_ratio * num_non_zeros)
     
        for name in masks:
            if num_remove[name]>0 and  "track" not in name and "running" not in name: 
                temp_weights = torch.where(masks[name].to(self.args.device) > 0, torch.abs(weights[name]),
                                        100000 * torch.ones_like(weights[name]))
                x, idx = torch.sort(temp_weights.view(-1).to(self.args.device))
                masks[name].view(-1)[idx[:num_remove[name]]] = 0
        return masks, num_remove



    def local_train(self, global_model, criterion, round=None, temparature=10, alpha=0.3, global_mask= None, neurotoxin_mask =None, updates_dict =None):
        """ Do a local training over the received global model, return the update """
        initial_global_model_params = parameters_to_vector([ global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        if self.id  <  self.args.num_corrupt:
            self.check_poison_timing(round)
        global_model.to(self.args.device)
        for name, param in global_model.named_parameters():
            self.mask[name] =self.mask[name].to(self.args.device)
            param.data = param.data * self.mask[name]
        if self.num_remove!=None:
            if self.id>=  self.args.num_corrupt or self.args.attack!="fix_mask" :
                gradient = self.screen_gradients(global_model)
                self.mask = self.update_mask(self.mask, self.num_remove, gradient)
        
        global_model.train()
        lr = self.args.client_lr* (self.args.lr_decay)**round
        optimizer = torch.optim.SGD(global_model.parameters(), lr=lr, weight_decay=self.args.wd)

        for _ in range(self.args.local_ep):
            for _, (inputs, labels) in enumerate(self.train_loader):
                optimizer.zero_grad()
                inputs, labels = inputs.to(device=self.args.device), \
                                 labels.to(device=self.args.device)
                outputs = global_model(inputs)
                minibatch_loss = criterion(outputs, labels)
                loss = minibatch_loss
                loss.backward()
                for name, param in global_model.named_parameters():
                    param.grad.data = self.mask[name].to(self.args.device) * param.grad.data
                optimizer.step()

           
        

        if self.id<  self.args.num_corrupt:
            if self.args.attack=="fix_mask":
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
            else:
                self.mask, self.num_remove = self.fire_mask(global_model.state_dict(), self.mask, round)

        else:
            self.mask, self.num_remove = self.fire_mask(global_model.state_dict(), self.mask, round) 
            
        with torch.no_grad():
            after_train = parameters_to_vector([ global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
            array_mask = parameters_to_vector([ self.mask[name].to(self.args.device) for name in global_model.state_dict()]).detach()
            self.update = ( array_mask *(after_train - initial_global_model_params))
            if "scale" in self.args.attack:
                logging.info("scale update for" + self.args.attack.split("_",1)[1] + " times")
                if self.id<  self.args.num_corrupt:
                    self.update=  int(self.args.attack.split("_",1)[1]) * self.update
        return self.update

