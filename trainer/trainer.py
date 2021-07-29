# commenting the original btad for unet based real/fake 
import torch
from utils.optimize import *
from .base_trainer import BaseTrainer
from pytorch_memlab import profile
from easydict import EasyDict as edict
import os.path as osp
from dataset import dataset
import  torch.optim as optim
from tqdm import tqdm
# import neptune
import math
from PIL import Image
import copy
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import torch.nn as nn

class Trainer(BaseTrainer):
    def __init__(self, model, config, writer):
        self.model = model

        self.config = config
        self.writer = writer
        
    def iter(self, batch):
        
        img, seg_label, _, _, name = batch
        # print(img.shape) 
        
        # seg_label = seg_label.long().cuda() # cross entropy 
        seg_label = seg_label.float().cuda() # bce or focal loss 
        b, c, h, w = img.shape
        # print(img.shape)

        seg_pred = self.model(img.cuda())
        # seg_pred = nn.DataParallel(self.model(img.cuda()))
        seg_pred = seg_pred.squeeze(dim=1) #for bce or focal loss 
        # print('yes')
        # print(seg_label.shape, seg_pred.shape)
        # seg_loss = F.binary_cross_entropy(seg_pred, seg_label)
        # print(seg_label.shape)
        # print('&&&&&&&&&&&&&&&&&&&&&&&&')
        # print(seg_pred.shape)
        # print('**********************')
        loss = WeightedFocalLoss(alpha= 0.75, gamma=3) # alpha ...gamma are hyper param ..have to decide later ## chaged from alpha =0.75, gamma = 2
        # loss = WeightedFocalLoss() #not workinjg ...cause the class imbalance issue..is major here...
        # print('****')
        seg_loss = loss(seg_pred, seg_label)
        # loss = CrossEntropy2d() # ce loss 
        # seg_loss = loss(seg_pred, seg_label)
        self.losses.seg_loss = seg_loss
        loss = seg_loss  
        loss.backward()

    def train(self):
        writer = SummaryWriter(comment="reak_fake_unet_chgfoclr_resize")

        if self.config.neptune:
            neptune.init(project_qualified_name='solacex/segmentation-DA')
            neptune.create_experiment(params=self.config, name=self.config['note'])

        if self.config.multigpu:       
            self.optim = optim.SGD(self.model.module.optim_parameters(self.config.learning_rate),
                          lr=self.config.learning_rate, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        else:
            # self.optim = optim.SGD(self.model.optim_parameters(self.config.learning_rate),
            #               lr=self.config.learning_rate, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
            self.optim = optim.SGD(self.model.parameters(),
                          lr=self.config.learning_rate, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
            # self.optim = optim.Adam(self.model.parameters(),
            #             lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        
        self.loader, _ = dataset.init_source_dataset(self.config)#, source_list=self.config.src_list)

        cu_iter = 0
        best_val_epoch_loss = float('inf')
        train_epoch_loss = 0
        print('Lets ride...')

        for epoch in range(self.config.epochs):
            epoch_infor = ('epoch = {:6d}/{:6d}, exp = {}'.format(epoch, int(self.config.epochs), self.config.note))
            
            print(epoch_infor)
            # print(len(self.loader))
            # print(self.config.learning_rate)

            for i_iter, batch in tqdm(enumerate(self.loader)):
                cu_iter +=1
                adjust_learning_rate(self.optim, cu_iter, self.config)
                self.optim.zero_grad()
                self.losses = edict({})
                losses = self.iter(batch)

                # print(self.config['model'])

                # print(self.losses['seg_loss'].item())
                # print(cu_iter)
                train_epoch_loss += self.losses['seg_loss'].item()
                print('train_iter_loss:', self.losses['seg_loss'].item())
                # valid_epoch_loss = self.validatebtad(i_iter)
                # print('validation_loss: ', valid_epoch_loss)
                self.optim.step() 

            train_epoch_loss /= len(iter(self.loader)) 
            # if cu_iter % self.config.print_freq ==0:
            #     self.print_loss(cu_iter)
            # if self.config.val and cu_iter % self.config.val_freq ==0 and cu_iter!=0:
                # miou = self.validate() 
            if self.config.val:

                loss_infor = 'train loss :{:.4f}'.format(train_epoch_loss)
                print(loss_infor)

                valid_epoch_loss = self.validatebtad(epoch)

                writer.add_scalars('Epoch_loss',{'train_rf_epoch_loss': train_epoch_loss,'valid_rf_epoch_loss':valid_epoch_loss},epoch)  

                if(valid_epoch_loss < best_val_epoch_loss):
                    best_val_epoch_loss = valid_epoch_loss
                    print('********************')
                    print('best_val_epoch_loss: ', best_val_epoch_loss)
                    print("MODEL UPDATED")
                    name = self.config['model'] + '.pth' # for the ce loss 
                    # name = 'acdc_tensor_focal.pth' # focal loss 
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


class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        # print(predict.shape)
        # print('***************')
        # print(target.shape)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss

class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2, ignore_label=255):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma
        self.ignore_label = ignore_label

    def forward(self, inputs, targets):
        # n, h, w = inputs.size()
        targets_mask = (targets >= 0) * (targets != self.ignore_label)
        targets = targets[targets_mask]
        inputs = inputs[targets_mask]
        BCE_loss = F.binary_cross_entropy(inputs, targets)
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        # print(at[targets==1])
        # # print(at)
        # print('**********')
        pt = torch.exp(-BCE_loss)
        # print(pt)
        # print('^^^^^^^^^^^^')
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()