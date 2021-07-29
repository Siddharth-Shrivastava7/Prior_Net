import torch
from utils.optimize import *
from .base_trainer import BaseTrainer
from pytorch_memlab import profile
from easydict import EasyDict as edict
import os.path as osp
from dataset import dataset
import  torch.optim as optim
from tqdm import tqdm
import neptune
import math
from PIL import Image
import copy
import torch.nn.functional as F
from tensorboardX import SummaryWriter

class Trainer(BaseTrainer):
    def __init__(self, model, config, writer):
        self.model = model

        self.config = config
        self.writer = writer
        
    def iter(self, batch):
        
        img, seg_label, _, _, name = batch
        
        seg_label = seg_label.long().cuda()
        b, c, h, w = img.shape
        # print(img.shape)

        seg_pred = self.model(img.cuda())
        # print(seg_label.shape, seg_pred.shape)
        seg_loss = F.cross_entropy(seg_pred, seg_label, ignore_index=255)
        self.losses.seg_loss = seg_loss
        loss = seg_loss  
        loss.backward()

    def train(self):
        writer = SummaryWriter(comment="exp_source_cityscape_b8_e1000")

        if self.config.neptune:
            neptune.init(project_qualified_name='solacex/segmentation-DA')
            neptune.create_experiment(params=self.config, name=self.config['note'])

        if self.config.multigpu:       
            self.optim = optim.SGD(self.model.module.optim_parameters(self.config.learning_rate),
                          lr=self.config.learning_rate, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        else:
            self.optim = optim.SGD(self.model.optim_parameters(self.config.learning_rate),
                          lr=self.config.learning_rate, momentum=self.config.momentum, weight_decay=self.config.weight_decay)

        self.loader, _ = dataset.init_source_dataset(self.config)#, source_list=self.config.src_list)

        cu_iter = 0
        best_val_epoch_loss = float('inf')
        train_epoch_loss = 0

        for epoch in range(self.config.epochs):
            epoch_infor = ('epoch = {:6d}/{:6d}, exp = {}'.format(epoch, int(self.config.epochs), self.config.note))
            
            print(epoch_infor)

            for i_iter, batch in tqdm(enumerate(self.loader)):
                # cu_iter +=1
                # adjust_learning_rate(self.optim, cu_iter, self.config)
                self.optim.zero_grad()
                self.losses = edict({})
                losses = self.iter(batch)

                # print(self.losses['seg_loss'].item())
                # print(cu_iter)
                train_epoch_loss += self.losses['seg_loss'].item()
                self.optim.step()

            train_epoch_loss /= len(iter(self.loader)) 
            # if cu_iter % self.config.print_freq ==0:
            #     self.print_loss(cu_iter)
            # if self.config.val and cu_iter % self.config.val_freq ==0 and cu_iter!=0:
                # miou = self.validate()
            if self.config.val:

                loss_infor = 'train loss :{:.4f}'.format(train_epoch_loss)
                print(loss_infor)

                valid_epoch_loss = self.validate2(epoch)

                writer.add_scalars('Epoch_loss',{'train_epoch_loss': train_epoch_loss,'valid_epoch_loss':valid_epoch_loss},epoch)  

                if(valid_epoch_loss < best_val_epoch_loss):
                    best_val_epoch_loss = valid_epoch_loss
                    print('********************')
                    print('best_val_epoch_loss: ', best_val_epoch_loss)
                    print("MODEL UPDATED")
                    name = 'modelval.pth'
                    torch.save(self.model.state_dict(), osp.join(self.config["snapshot"], name))
                    
                self.model = self.model.train()

        writer.export_scalars_to_json("./all_scalars.json")
        writer.close()

        if self.config.neptune:
            neptune.stop()
                
    def resume(self):
        self.tea = copy.deepcopy(self.model)
        self.round_start = self.config.round_start #int(math.ceil(iter_num/self.config.num_steps) -1 )
        print('Resume from Round {}'.format(self.round_start))
        if self.config.lr_decay == 'sqrt':
            self.config.learning_rate = self.config.learning_rate/((math.sqrt(2))**self.round_start)

    def save_best(self, name):
        name = str(name)
        if 'pth' not in name:
            name = name +'.pth'
        torch.save(self.model.state_dict(), osp.join(self.config["snapshot"], name))
    def save_model(self, iter, rep_teacher=False):
        tmp_name = '_'.join((self.config.source, str(iter))) + '.pth'
        torch.save(self.model.state_dict(), osp.join(self.config['snapshot'], tmp_name))

