import torch
import os.path as osp
# import neptune
import torch.nn as nn
# import neptune 
from dataset import dataset_2
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
class BaseTrainer(object):
    def __init__(self, models, optimizers, loaders, up_s, up_t, config,  writer):
        self.model = models
        self.optim = optimizers
        self.loader = loaders
        self.config = config
        self.output = {}
        self.up_src = up_s
        self.up_tgt = up_t
        self.writer = writer

    def forward(self):
        pass
    def backward(self):
        pass

    def iter(self):
        pass

        
    def train(self):
        for i_iter in range(self.config.num_steps):
            losses = self.iter(i_iter)
            if i_iter==0 and self.config.neptune:
                neptune.init(project_qualified_name='solacex/segmentation-DA')
                neptune.create_experiment(params=self.config, name=self.config['note'])
            if i_iter % self.config.print_freq ==0:
                self.print_loss(i_iter)
            if i_iter % self.config.save_freq ==0 and i_iter != 0:
                self.save_model(i_iter)
            if self.config.val and i_iter % self.config.val_freq ==0 and i_iter!=0:
                self.validate()
        neptune.stop()

    def save_model(self, iter):
        tmp_name = '_'.join((self.config.source, str(iter))) + '.pth'
        torch.save(self.model.state_dict(), osp.join(self.config['snapshot'], tmp_name))

    def print_loss(self, iter):
        iter_infor = ('iter = {:6d}/{:6d}, exp = {}'.format(iter, self.config.num_steps, self.config.note))
        to_print = ['{}:{:.4f}'.format(key, self.losses[key].item()) for key in self.losses.keys()]
        loss_infor = '  '.join(to_print)
        if self.config.screen:
            print(iter_infor +'  '+ loss_infor)
        if self.config.neptune:
            for key in self.losses.keys():
                neptune.send_metric(key, self.losses[key].item())
        if self.config.tensorboard and self.writer is not None:
            for key in self.losses.keys():
                self.writer.add_scalar('train/'+key, self.losses[key], iter)

    def validate(self):
        self.model = self.model.eval()
        testloader = dataset_2.init_test_dataset(self.config, self.config.target_val, set='val')
        interp = nn.Upsample(size=(1080, 1920), mode='bilinear', align_corners=True)
        union = torch.zeros(self.config.num_classes, 1,dtype=torch.float).cuda().float()
        inter = torch.zeros(self.config.num_classes, 1, dtype=torch.float).cuda().float()
        preds = torch.zeros(self.config.num_classes, 1, dtype=torch.float).cuda().float()
        with torch.no_grad():
            for index, batch in tqdm(enumerate(testloader)):
                image, label, _, _, name = batch
                output =  self.model(image.cuda())
                label = label.cuda()
                output = interp(output).squeeze()
                C, H, W = output.shape
                Mask = (label.squeeze())<C

                pred_e = torch.linspace(0,C-1, steps=C).view(C, 1, 1)
                pred_e = pred_e.repeat(1, H, W).cuda()
                pred = output.argmax(dim=0).float()
                pred_mask = torch.eq(pred_e, pred).byte()
                pred_mask = pred_mask*Mask.byte()

                label_e = torch.linspace(0,C-1, steps=C).view(C, 1, 1)
                label_e = label_e.repeat(1, H, W).cuda()
                label = label.view(1, H, W)
                label_mask = torch.eq(label_e, label.float()).byte()
                label_mask = label_mask*Mask.byte()

                tmp_inter = label_mask+pred_mask.byte()
                cu_inter = (tmp_inter==2).view(C, -1).sum(dim=1, keepdim=True).float()
                cu_union = (tmp_inter>0).view(C, -1).sum(dim=1, keepdim=True).float()
                cu_preds = pred_mask.view(C, -1).sum(dim=1, keepdim=True).float()

                union+=cu_union
                inter+=cu_inter
                preds+=cu_preds

            iou = inter/union
            acc = inter/preds
            if C==16:
                iou = iou.squeeze()
                class13_iou = torch.cat((iou[:3], iou[6:]))
                class13_miou = class13_iou.mean().item()
                print('13-Class mIoU:{:.2%}'.format(class13_miou))
            mIoU = iou.mean().item()
            mAcc = acc.mean().item()
            iou = iou.cpu().numpy()
            print('mIoU: {:.2%} mAcc : {:.2%} '.format(mIoU, mAcc))
            if self.config.neptune:
                neptune.send_metric('mIoU', mIoU)
                neptune.send_metric('mAcc', mAcc)
        return mIoU

    
    def validate2(self, count):
        self.model = self.model.eval()
        total_loss = 0
        testloader = dataset_2.init_test_dataset(self.config, self.config.target_val, set='val')
        interp = nn.Upsample(size=(1080,1920), mode='bilinear', align_corners=True)
        for i_iter, batch in tqdm(enumerate(testloader)):
            img, seg_label, _, _, name = batch
            seg_label = seg_label.long().cuda()
            b, c, h, w = img.shape
            # print(img.shape)
            seg_pred = self.model.forward(img.cuda())
            # print(seg_pred.shape, seg_label.shape)
            seg_pred = interp(seg_pred)
            seg_loss = F.cross_entropy(seg_pred, seg_label, ignore_index=255)
            total_loss += seg_loss.item()
        total_loss /= len(iter(testloader))
        print('---------------------')
        print('Validation seg loss: {} at iter {}'.format(total_loss,count))
        return total_loss
    
    # def validatebtad(self, count):
    #     self.model = self.model.eval()
    #     total_loss = 0
    #     testloader = dataset.init_test_datasetbtad(self.config, self.config.target, set='val')
    #     interp = nn.Upsample(size=(1080,1920), mode='bilinear', align_corners=True)
    #     union = torch.zeros(self.config.num_classes, 1,dtype=torch.float).cuda().float()
    #     inter = torch.zeros(self.config.num_classes, 1, dtype=torch.float).cuda().float()
    #     preds = torch.zeros(self.config.num_classes, 1, dtype=torch.float).cuda().float()
    #     with torch.no_grad():
    #         for index, batch in tqdm(enumerate(testloader)):
    #             image, label, _, _, name = batch
    #             output =  self.model(image.cuda())
    #             label = label.cuda()
    #             output = interp(output).squeeze()
    #             C, H, W = output.shape
    #             Mask = (label.squeeze())<C

    #             pred_e = torch.linspace(0,C-1, steps=C).view(C, 1, 1)
    #             pred_e = pred_e.repeat(1, H, W).cuda()
    #             pred = output.argmax(dim=0).float()
    #             pred_mask = torch.eq(pred_e, pred).byte()
    #             pred_mask = pred_mask*Mask.byte()

    #             label_e = torch.linspace(0,C-1, steps=C).view(C, 1, 1)
    #             label_e = label_e.repeat(1, H, W).cuda()
    #             label = label.view(1, H, W)
    #             label_mask = torch.eq(label_e, label.float()).byte()
    #             label_mask = label_mask*Mask.byte()

    #             tmp_inter = label_mask+pred_mask.byte()
    #             cu_inter = (tmp_inter==2).view(C, -1).sum(dim=1, keepdim=True).float()
    #             cu_union = (tmp_inter>0).view(C, -1).sum(dim=1, keepdim=True).float()
    #             cu_preds = pred_mask.view(C, -1).sum(dim=1, keepdim=True).float()

    #             union+=cu_union
    #             inter+=cu_inter
    #             preds+=cu_preds

    #         iou = inter/union
    #         acc = inter/preds
    #         if C==16:
    #             iou = iou.squeeze()
    #             class13_iou = torch.cat((iou[:3], iou[6:]))
    #             class13_miou = class13_iou.mean().item()
    #             print('13-Class mIoU:{:.2%}'.format(class13_miou))
    #         mIoU = iou.mean().item()
    #         mAcc = acc.mean().item()
    #         iou = iou.cpu().numpy()
    #         print('mIoU: {:.2%} mAcc : {:.2%} '.format(mIoU, mAcc))
    #         if self.config.neptune:
    #             neptune.send_metric('mIoU', mIoU)
    #             neptune.send_metric('mAcc', mAcc)
                
    #     for i_iter, batch in tqdm(enumerate(testloader)):
    #         img, seg_label, _, _, name = batch
    #         seg_label = seg_label.long().cuda()
    #         b, c, h, w = img.shape
    #         # print(img.shape)
    #         seg_pred = self.model(img.cuda())
    #         # print(seg_pred.shape, seg_label.shape)
    #         seg_pred = interp(seg_pred)
    #         seg_loss = F.cross_entropy(seg_pred, seg_label, ignore_index=255)
    #         total_loss += seg_loss.item()
    #     total_loss /= len(iter(testloader))
    #     print('---------------------')
    #     print('Validation seg loss: {} at iter {}'.format(total_loss,count))
    #     return total_loss

    def validatebtad(self, epoch):
        self.model = self.model.eval() 
        total_loss = 0
        testloader = dataset_2.init_test_dataset(self.config, self.config.target, set='val')
        for i_iter, batch in tqdm(enumerate(testloader)):
            img, seg_label, _, _, name = batch
            # print(img.shape)
            # print('******')
            # print(seg_label.shape)
            # seg_label = seg_label.long().cuda() # ce loss
            seg_label = seg_label.float().cuda() # bce or focal loss 
            b, c, h, w = img.shape
            # print(img.shape)
            seg_pred = self.model(img.cuda())
            seg_pred = seg_pred.squeeze(dim=1) # for bce or focal loss 
            # print(seg_pred.shape, seg_label.shape)
            # seg_loss = F.binary_cross_entropy(seg_pred, seg_label)
            # loss = CrossEntropy2d() 
            # seg_loss = loss(seg_pred, seg_label)
            loss = WeightedFocalLoss(alpha=0.75, gamma=2)
            seg_loss = loss(seg_pred, seg_label)
            total_loss += seg_loss.item()
        total_loss /= len(iter(testloader))
        print('---------------------')
        print('Validation seg loss: {} at epoch {}'.format(total_loss,epoch))
        return total_loss
    

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