# commenting the original btad for unet based real/fake 
import os, sys
import GPUtil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from dataset.dataset import *  # init_dataset
from model.model_builder_btad import init_model
from model import *
from init_config import *
from easydict import EasyDict as edict
import sys
from trainer.btad_trainer_2 import Trainer
import trainer
import time, datetime
import copy
import numpy as np 
import random

# .. yml file .. chg

def main():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    cudnn.enabled = True
    cudnn.benchmark = True
#    torch.backends.cudnn.deterministic = True
    config, writer = init_config("config/so_configmodbtad.yml", sys.argv)
    # if config.source=='synthia':
    #     config.num_classes=16
    # elif config.source == 'city_rf':
    #     # print('***********************')
    #     config.num_classes = 2
    # else:
    #     config.num_classes=19

    model = init_model(config)

    trainer = Trainer(model, config, writer)

    trainer.train()


if __name__ == "__main__":
    start = datetime.datetime(2020, 1, 22, 23, 00, 0)
    print("wait")
    while datetime.datetime.now() < start:
        time.sleep(1)
    main()
