# commenting the original btad for unet based real/fake 
from collections import OrderedDict
from .sync_batchnorm import convert_model
import torch
from .DeeplabV2 import *
from .Unet import *

def freeze_bn(net):
    for module in net.modules():
        if isinstance(module, torch.nn.modules.BatchNorm2d):
            for i in module.parameters():
                i.requires_grad = False

def release_bnad(net):
    for module in net.modules():
        if isinstance(module, torch.nn.modules.BatchNorm2d):
            for i in module.parameters():
                i.requires_grad = True
        else:
            for i in module.parameters():
                i.requires_grad = False            

def init_model(cfg):

    # model = Res_Deeplab(num_classes = cfg.num_classes).cuda()
    # model = Unet_model(num_classes= cfg.num_classes).cuda()
    # model = Unet_Discriminator(resolution = 256).cuda()
    model = UNet_mod().cuda() 
    # model = UNet_mod_2(n_class=2).cuda()

    # if cfg.fixbn:
    #     freeze_bn(model)
    # else:
    #     release_bnad(model)


#     if cfg.model=='deeplab' and cfg.init_weight != 'None':
#         params = torch.load(cfg.init_weight)
#         print('Model restored with weights from : {}'.format(cfg.init_weight))
#         if 'init-' in cfg.init_weight and cfg.model=='deeplab':
#             new_params = model.state_dict().copy()
#             for i in params:
#                 i_parts = i.split('.')
#                 if not i_parts[1] == 'layer5':
#                     new_params['.'.join(i_parts[1:])] = params[i]
#             model.load_state_dict(new_params, strict=True)

#         else:
#             new_params = model.state_dict().copy()
#             for i in params:
#                 if 'module' in i:
#                     i_ = i.replace('module.', '')
#                     new_params[i_] = params[i]
#                 else:
#                     new_params[i] = params[i]
# #                i_parts = i.split('.')[0]
#             model.load_state_dict(new_params, strict=True)
# #            model.load_state_dict(params, strict=True)

    if cfg.restore_from != 'None':
        params = torch.load(cfg.restore_from)
        model.load_state_dict(params)
        print('----------Model initialize with weights from-------------: {}'.format(cfg.restore_from))

    if cfg.multigpu:
        model = convert_model(model)
        model = nn.DataParallel(model)
    if cfg.train:
        model.train().cuda()
        print('Mode --> Train')
    else:
        model.eval().cuda()
        print('Mode --> Eval')
    
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad)) 105344 # for only batch normalisation
    
    return model

