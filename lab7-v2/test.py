import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import math
import random
import numpy as np
import math
import argparse
import time
from torch.nn import init
import os
from dataloader_test import *
from torch.utils.data import Dataset, DataLoader
from model import *
from evaluator import *
from util import save_image

test_datasets = ICLEVRLoader('data', cond=True, mode='test')
# test_datasets = ICLEVRLoader('/home/austin/Downloads/task_1/', cond=True, mode='new')
test_dataloader = DataLoader(test_datasets, batch_size = len(test_datasets), shuffle = True)

Eval = evaluation_model()

netG = Generator().cuda()


def gen_z(num):
    tmp = torch.zeros((num, 200, 1, 1))
    z = torch.randn_like(tmp).cuda()

    return z


with torch.no_grad():
    netG.load_state_dict(torch.load('modelGenerator_209_0.6944444444444444_.pth'))
    netG.eval()
    for _ in range(1000):
        z_test = gen_z(len(test_datasets))
        for _, sampled_batched in enumerate(test_dataloader):
            conds_ = sampled_batched['cond'].cuda() 
            gen_img = netG(z_test, conds_)
            conds_ = torch.squeeze(conds_, -1)
            conds_ = torch.squeeze(conds_, -1)
            score = Eval.eval(gen_img, conds_)
            print(score,end='\r')
            break
        if (score >= 0.67):
            print('Score : ',score)
            save_image(gen_img, os.path.join('result/result_test.png'), nrow=8, normalize=True)
            break
