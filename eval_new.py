import os, sys
import argparse
import numpy as np
import torch
from model.DeeplabV2 import *#Res_Deeplab
from model.Unet import * 
from collections import OrderedDict

from torch.utils import data
import torch.nn as nn
import os.path as osp
import yaml
# from utils.logger import Logger 
from dataset.dataset import *
from easydict import EasyDict as edict
from tqdm import tqdm
from PIL import Image
import json
import torchvision
import cv2


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--num-classes", type=int, default=19)
    parser.add_argument("--frm", type=str, default=None)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--dataset", type=str, default='darkzurich_val')
    parser.add_argument("--single", action='store_true')
    parser.add_argument("--model", default='deeplab')
    return parser.parse_args()

def print_iou(iou, acc, miou, macc):
    for ind_class in range(iou.shape[0]):
        print('===> {0:2d} : {1:.2%} {2:.2%}'.format(ind_class, iou[ind_class, 0].item(), acc[ind_class, 0].item()))
    print('mIoU: {:.2%} mAcc : {:.2%} '.format(miou, macc))

def compute_iou(model, testloader, args):
    model = model.eval()

    interp = nn.Upsample(size=(1080,1920), mode='bilinear', align_corners=True)   # dark_zurich -> (1080,1920)
    union = torch.zeros(args.num_classes, 1,dtype=torch.float).cuda().float()
    inter = torch.zeros(args.num_classes, 1, dtype=torch.float).cuda().float()
    preds = torch.zeros(args.num_classes, 1, dtype=torch.float).cuda().float()
    # extra 
    gts = torch.zeros(args.num_classes, 1, dtype=torch.float).cuda().float()
    # 2nd best 
    # totalp = 0
    # Tp = 0  
    with torch.no_grad():
        for index, batch in tqdm(enumerate(testloader)):
            # print('******************') 
            image, label, edge, _, name = batch
#            edge = F.interpolate(edge.unsqueeze(0), (512, 1024)).view(1,512,1024)``
            # print(name)
            if name[0].find('dannet_pred')==-1: 
                continue
            # print(name)
            # output =  model(image.cuda())
            label = label.cuda()
            # print('label shape:{} output shape:{}'.format(label.shape, output.shape))
            # output = interp(output).squeeze()
            # output = output.squeeze()
            # save_pred(output, './save/dark_zurich_val/btad', args.dataset +str(index)+'.png') # org
            # print(name[0])
            name =name[0].split('/')[-1]
            # save_pred(output, '../scratch/data/try', name)  # current org # now not save

            output = image.cuda() 
            output = output.squeeze()
            C = 2
            # print(output.shape)
            # print('**********')
            H, W = output.shape
            # C, H, W = output.shape # original
            # print('[*****]')
            # print(C)
            # print(torch.unique(output))
            # print(torch.unique(torch.argmax(output, dim = 0)))
            

            #########################################################################original
            Mask = (label.squeeze())<C  # it is ignoring all the labels values equal or greater than 2 #(1080, 1920) 
            pred_e = torch.linspace(0,C-1, steps=C).view(C, 1, 1)  
            pred_e = pred_e.repeat(1, H, W).cuda() 
            # pred = output.argmax(dim=0).float()
            pred = output.float()
            pred_mask = torch.eq(pred_e, pred).byte() 
            pred_mask = pred_mask*Mask
            # print(Mask.shape) #torch.Size([1080, 1920])
            # print(pred_mask.shape) #torch.Size([2, 1080, 1920]) 

            label_e = torch.linspace(0,C-1, steps=C).view(C, 1, 1)
            label_e = label_e.repeat(1, H, W).cuda()
            label = label.view(1, H, W)
            label_mask = torch.eq(label_e, label.float()).byte()
            label_mask = label_mask*Mask

            tmp_inter = label_mask+pred_mask
            cu_inter = (tmp_inter==2).view(C, -1).sum(dim=1, keepdim=True).float()
            cu_union = (tmp_inter>0).view(C, -1).sum(dim=1, keepdim=True).float()
            cu_preds = pred_mask.view(C, -1).sum(dim=1, keepdim=True).float()
            #extra
            cu_gts = label_mask.view(C, -1).sum(dim=1, keepdim=True).float()
            gts += cu_gts
            union+=cu_union
            inter+=cu_inter
            preds+=cu_preds
            #########################################################################origninal

            ########################################################################with2ndbest
            # Mask = (label.squeeze())<C
            # pred_e = torch.linspace(0,C-1, steps=C).view(C, 1, 1)
            # pred_e = pred_e.repeat(1, H, W).cuda()
            # pred = output.argmax(dim=0).float()  # most confident pred 
            # pred2 = output.topk(2,dim=0)[1][1].float()  # 2nd most confi   
            # pred_mask = torch.eq(pred_e, pred).byte()
            # pred2_mask = torch.eq(pred_e, pred2).byte()
            # pred_mask = pred_mask*Mask
            # pred2_mask = pred2_mask*Mask

            # label_e = torch.linspace(0,C-1, steps=C).view(C, 1, 1)
            # label_e = label_e.repeat(1, H, W).cuda()
            # label = label.view(1, H, W)
            # label_mask = torch.eq(label_e, label.float()).byte()
            # label_mask = label_mask*Mask

            # pred_maskb = torch.logical_or(pred_mask, pred2_mask) # for combining both 1st most and 2 most confident pred       

            # tmp_interb = label_mask+pred_maskb  # for getting top 2 acc 
            # tmp_inter = label_mask + pred_mask  
            # tmp_inter2 = label_mask + pred2_mask   

            # cu_inter = (tmp_inter2==2).view(C, -1).sum(dim=1, keepdim=True).float() # overall including both predictions

            # #####ignore this 
            # cu_union = (tmp_inter2>0).view(C, -1).sum(dim=1, keepdim=True).float() 
            # cu_preds = pred2_mask.view(C, -1).sum(dim=1, keepdim=True).float() 
            # ##### ignore this 

            # Tp += sum((tmp_interb==2).view(C, -1).sum(dim=1, keepdim=True)).float()
            # totalp += H*W      

            # # print(pred_mask.shape)
            # union+=cu_union
            # inter+=cu_inter
            # preds+=cu_preds
            ##########################################################################with2ndbest
        
        ###########original
        iou = inter/union
        acc = inter/preds
        mIoU = iou.mean().item()
        mAcc = acc.mean().item()
        print('*********')
        print(gts)
        print_iou(iou, acc, mIoU, mAcc)
        ##################
        
        # print('*****************************')
        # mean_acc = Tp/totalp 
        # print('mean_acc: ', mean_acc)
        # print('*****************************')

        # iou = inter/union
        # acc = inter/preds
        # mIoU = iou.mean().item()
        # mAcc = acc.mean().item()
        # print_iou(iou, acc, mIoU, mAcc)    

        return iou, mIoU, acc, mAcc

def label_img_to_color(img):
    # label_to_color = {
    #     0: [128, 64,128],
    #     1: [244, 35,232],
    #     2: [ 70, 70, 70],
    #     3: [102,102,156],
    #     4: [190,153,153],
    #     5: [153,153,153],
    #     6: [250,170, 30],
    #     7: [220,220,  0],
    #     8: [107,142, 35],
    #     9: [152,251,152],
    #     10: [ 70,130,180],
    #     11: [220, 20, 60],
    #     12: [255,  0,  0],
    #     13: [  0,  0,142],
    #     14: [  0,  0, 70],
    #     15: [  0, 60,100],
    #     16: [  0, 80,100],
    #     17: [  0,  0,230],
    #     18: [119, 11, 32],
    #     19: [0,  0, 0]
    #     }
    # # with open('./dataset/cityscapes_list/info.json') as f:
    #     data = json.load(f)

    label_to_color = {
        0: [0, 0, 0],
        1: [255,255,255]
    }

    img_height, img_width = img.shape

    img_color = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    for row in range(img_height):
        for col in range(img_width):
            label = img[row][col]
            img_color[row, col] = np.array(label_to_color[label])
            # img_color[row][col] = np.asarray(data['palette'][label])
    return img_color


def save_fake(model, testloader):
    model = model.eval()
    # interp = nn.Upsample(size=(1024,2048), mode='bilinear', align_corners=True)
    with torch.no_grad():
        for index, batch in tqdm(enumerate(testloader)):
            image, label, edge, _, name = batch
            output =  model(image.cuda())
            # output = interp(output).squeeze()
            output = output.squeeze()
            name = name[0].split("/")[-1]
            # print(name)
            save_pred(output, '../scratch/saved_models/CCM/save/result/rf_city_dzval', name)
    return


def save_pred(pred, direc, name):
    # palette = get_palette(256)
    pred = pred.cpu().numpy()
    # print(pred.shape)
    # pred = np.asarray(np.argmax(pred, axis=0), dtype=np.uint8)   ##### original
    # pred = np.asarray(np.argsort(pred, axis= 0)[-2], dtype = np.uint8)  ############ 2nd best prediction
    
    # if thresholding for binary segmentation 
    pred[pred<0.5] = 0
    pred[pred>=0.5] = 1
    
    # pred = np.asarray(np.argmax(pred, axis=0))
    # print(pred.shape)
    label_img_color = label_img_to_color(pred)
    # print(label_img_color.dtype)
    # print(label_img_color.shape)
    # print(np.unique(label_img_color))
    # cv2.imwrite(osp.join(direc,name), label_img_color)
    im = Image.fromarray(label_img_color) # use always PIL or other library .. try to avoid cv2.. but if no other option then ok
    # im.save(osp.join(direc,name))
    im.save(osp.join(direc,name))
    return 
    # print('img saved!')

    # img = np.zeros((pred.shape[0],pred.shape[1],3))

    # print(np.unique(pred))
    # print(pred.shape)
    # print(plabel.shape)
    # print(img.shape)
    # cv2.imwrite("pred.png", img)
    # output_im = Image.fromarray(pred)
    # output_im.putpalette(palette)
    # output_im.save('pred.png')
         
    # print((plabel==i).nonzero()) #work 
    # print(plabel==i)

    # img = torch.tensor(img)
    # img = img.permute(2, 0, 1).unsqueeze(dim = 0)
    # print(img.shape) 
    # torchvision.utils.save_image(img,osp.join(direc,name)) 

    # img = img.reshape(3,pred.shape[0],-1)
    # print(img.shape)
    # print(np.unique(img))
    # unique, counts = np.unique(img, return_counts=True)
    # print(dict(zip(unique, counts)))
    # img = Image.fromarray((img * 255).astype(np.uint8))
    # img.save(osp.join(direc,name))


def main():
    args = get_arguments()
    with open('./config/so_configmodbtad_2.yml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = edict(cfg)
    cfg.num_classes=args.num_classes
    if args.single:
        #from model.fuse_deeplabv2 import Res_Deeplab
        if args.model=='deeplab':
            # print('*********************************')
            model = Res_Deeplab(num_classes=args.num_classes).cuda()
        elif args.model == 'unet':
            # print(')))))))))))))))))))))))))))')
            model = UNet(n_class=2).cuda()
        else:
            model = FCN8s(num_classes = args.num_classes).cuda() 

        # model = nn.DataParallel(model)
        model.load_state_dict(torch.load(args.frm)) #original
        # model.load_state_dict(torch.load(args.frm,strict=False))

        # original saved file with DataParallel
        # state_dict = torch.load(args.frm)
        # # create new OrderedDict that does not contain `module.`
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     name = k[7:] # remove `module.`
        #     new_state_dict[name] = v
        # # load params
        # model.load_state_dict(new_state_dict)

        model.eval().cuda()
        testloader  = init_test_dataset(cfg, args.dataset, set='val')
        # print(len(testloader))
        # print('****************************************************')
        # save_fake(model, testloader)
        iou, mIoU, acc, mAcc = compute_iou(model, testloader, args) # original
        return

    # sys.stdout = Logger(osp.join(cfg['result'], args.frm+'.txt'))

    # best_miou = 0.0
    # best_iter = 0
    # best_iou = np.zeros((args.num_classes, 1))

   
    # for i in range(args.start, 25):
    #     model_path = osp.join(cfg['snapshot'], args.frm, 'GTA5_{0:d}.pth'.format(i*2000))# './snapshots/GTA2Cityscapes/source_only/GTA5_{0:d}.pth'.format(i*2000)
    #     model = Res_Deeplab(num_classes=args.num_classes)
    #     #model = nn.DataParallel(model)

    #     model.load_state_dict(torch.load(model_path))
    #     model.eval().cuda()
    #     testloader = init_test_dataset(cfg, args.dataset, set='train') 

    #     iou, mIoU, acc, mAcc = compute_iou(model, testloader)

    #     print('Iter {}  finished, mIoU is {:.2%}'.format(i*2000, mIoU))

if __name__ == '__main__':
    main()
