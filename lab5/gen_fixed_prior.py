import argparse
import itertools
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import bair_robot_pushing_dataset
from models.lstm import gaussian_lstm, lstm
from models.vgg_64 import vgg_decoder, vgg_encoder
from utils import init_weights, kl_criterion, plot_pred, plot_rec, finn_eval_seq, mse_metric


import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
    parser.add_argument('--batch_size', default=12, type=int, help='batch size')
    parser.add_argument('--log_dir', default='./logs/fp', help='base directory to save logs')
    parser.add_argument('--model_dir', default='result', help='base directory to save logs')
    parser.add_argument('--data_root', default='./data/processed_data', help='root directory for data')
    parser.add_argument('--optimizer', default='adam', help='optimizer to train with') #adam
    parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--epoch_size', type=int, default=300, help='epoch size') #600
    ###
    parser.add_argument('--tfr', type=float, default=1.0, help='teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_start_decay_epoch', type=int, default=120, help='The epoch that teacher forcing ratio become decreasing')
    parser.add_argument('--tfr_lower_bound', type=float, default=0.4, help='The lower bound of teacher forcing ratio for scheduling teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_decay_step', type=float, default=0, help='The decay step size of teacher forcing ratio (0 ~ 1)')
    ###
    parser.add_argument('--kl_anneal_cyclical', default=False, action='store_true', help='use cyclical mode')
    parser.add_argument('--kl_anneal_ratio', type=float, default=0.5, help='The decay ratio of kl annealing')
    parser.add_argument('--kl_anneal_cycle', type=int, default=5, help='The number of cycle for kl annealing (if use cyclical mode)')
    ###
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
    parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
    parser.add_argument('--n_eval', type=int, default=12, help='number of frames to predict at eval time')
    parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
    parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--cond_dim', type=int, default=7, help='dimensionality of cond')
    parser.add_argument('--z_dim', type=int, default=64, help='dimensionality of z_t')
    parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
    parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading threads')
    parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
    parser.add_argument('--cuda', default=True, action='store_true')  
    parser.add_argument('--visualize_beta', type=float, default=0)  
    args = parser.parse_args()
    return args

def plot_psnr(x, epoch):
    plt.plot(range(len(x)), x)           
    plt.xlabel('Iterations')
    plt.ylabel('PSNR')
    plt.title("PSNR curve")
    plt.savefig('PSNR_{a}.png'.format(a = epoch))
    plt.close()

def plot_result(KLD, MSE, LOSS, PSNR, BETA, TFR, epoch, args):

    fig=plt.figure()
    ratio = plt.subplot()
    value = ratio.twinx()

    l1, = ratio.plot(BETA, color='red', linestyle='dashed')

    l2, = ratio.plot(TFR, color='orange', linestyle='dashed')

    l3, = value.plot(KLD, color='blue')

    l4, = value.plot(MSE, color='green')

    l5, = value.plot(LOSS, color='cyan')

    x_sparse = np.linspace(0, epoch, np.size(PSNR))
    l6 = value.scatter(x_sparse, PSNR, color='yellow')

    
    ratio.set_xlabel('Iterations')
    ratio.set_ylabel("ratio")
    value.set_ylabel('Loss')
    plt.title("Training loss / ratio curve")
    plt.legend([l1, l2, l3, l4, l5, l6], ["kl_beta", "tfr", "KLD", "mse", "loss", "PSNR"])
    plt.savefig('plot_{a}.png'.format(a = epoch))  
    plt.close()
    # plt.show()

def pred(x, cond, modules, args, device):

    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    modules['posterior'].hidden = modules['posterior'].init_hidden()
    x = x.float()


    gen_seq = []
    x_in = x[:, 0]
    for i in range(1, args.n_eval):

        c = cond[:,i,:].float()
        # h = modules['encoder'](x[:,i-1])
        h = modules['encoder'](x_in)
        
        if args.last_frame_skip or i < args.n_past:   
            h, skip = h
        else:
            h, _ = h
        h = h.detach()
        


        if i < args.n_past:
            h_target = modules['encoder'](x[:,i])[0].detach()
            _, z_t, _ = modules['posterior'](h_target)
        else:
            z_t = torch.cuda.FloatTensor(args.batch_size, args.z_dim).normal_()


        if i < args.n_past:
            modules['frame_predictor'](torch.cat([h, z_t, c], 1))
            x_in = x[:, i]
        else:
            h = modules['frame_predictor'](torch.cat([h, z_t, c], 1)).detach()
            x_in = modules['decoder']([h, skip])
            gen_seq.append(x_in.data.cpu().numpy())
    gen_seq = torch.tensor(np.array(gen_seq))
    gen_seq = gen_seq.permute(1,0,2,3,4)
    return gen_seq




            

def main():
    args = parse_args()
    if args.cuda:
        assert torch.cuda.is_available(), 'CUDA is not available.'
        device = 'cuda'
    else:
        device = 'cpu'
    
    assert args.n_past + args.n_future <= 30 and args.n_eval <= 30
    assert 0 <= args.tfr and args.tfr <= 1
    assert 0 <= args.tfr_start_decay_epoch 
    assert 0 <= args.tfr_decay_step and args.tfr_decay_step <= 1

    
    
    saved_model = torch.load('%s/cvae_fp_cycle_model.pth' % args.model_dir)
    optimizer = args.optimizer
    niter = args.niter
    args = saved_model['args']
    args.optimizer = optimizer
    start_epoch = saved_model['last_epoch']



    print("Device: ", device)
    print("Random Seed: ", args.seed)
    if args.kl_anneal_cyclical:
        print("KL method: cyclical")
    else:
        print("KL method: monotonic")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(args)

    # ------------ build the models  --------------
    frame_predictor = saved_model['frame_predictor']
    posterior = saved_model['posterior']
    decoder = saved_model['decoder']
    encoder = saved_model['encoder']
    # --------- transfer to device ------------------------------------
    frame_predictor.to(device)
    posterior.to(device)
    encoder.to(device)
    decoder.to(device)

    # --------- load a dataset ------------------------------------
    print("test data:")
    test_data = bair_robot_pushing_dataset(args, 'test')
    test_loader = DataLoader(test_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)
    test_iterator = iter(test_loader)

    best_val_psnr = 0

    #################### EVALUATE #########################
    frame_predictor.eval()
    encoder.eval()
    decoder.eval()
    posterior.eval()
    psnr_list = []
    pred_list = []
    gt_list = []
    for i in range(len(test_data) // args.batch_size):
        try:
            test_seq, test_cond = next(test_iterator)
        except StopIteration:
            test_iterator = iter(test_iterator)
            test_seq, test_cond = next(test_iterator)
        
        test_seq = test_seq.to(device)
        test_cond = test_cond.to(device)
        pred_seq = pred(test_seq, test_cond, saved_model, args, device)
        _, _, psnr = finn_eval_seq(test_seq[:, args.n_past:].cpu(), pred_seq.cpu().numpy())
        val_psnr = np.mean(psnr)
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
        print("PSNR: {now} / Best PSNR: {record}".format(now = val_psnr, record = best_val_psnr))
        psnr_list.append(psnr)
        # plot
        # for j in range(10):
        #     gt_seq = test_seq[:, args.n_past:].cpu().numpy()


        #     img_pred = (np.transpose(pred_seq[1,j,:,:], (1, 2, 0)) + 1) * 255.0
        #     img_gt = (np.transpose(gt_seq[1,j,:,:], (1, 2, 0)) + 1) * 255.0
        #     img_compare = np.concatenate((img_gt, img_pred), axis = 1)

        #     pred_list.append(img_pred)
        #     gt_list.append(img_gt)

        #     data_pred = Image.fromarray(np.uint8(img_pred))
        #     data_gt = Image.fromarray(np.uint8(img_gt))
        #     data_compare = Image.fromarray(np.uint8(img_compare))

        #     data_pred.save('./result_seq/{dir}/{number}_psnr.png'.format(dir = i, number = j+2))
        #     data_gt.save('./result_seq/{dir}/{number}_gt.png'.format(dir = i, number = j+2))
        #     data_compare.save('./result_seq/{dir}/compare_{number}.png'.format(dir = i, number = j+2))
        #     if j == 9:
        #         pred_list = np.concatenate(pred_list, axis = 1)
        #         gt_list = np.concatenate(gt_list, axis = 1)
        #         compare = np.concatenate((gt_list,pred_list), axis = 0)
        #         final_pred = Image.fromarray(np.uint8(pred_list))
        #         final_gt = Image.fromarray(np.uint8(gt_list))
        #         final_compare = Image.fromarray(np.uint8(compare))
        #         final_pred.save('./result_seq/{dir}/final_psnr.png'.format(dir = i))
        #         final_gt.save('./result_seq/{dir}/final_gt.png'.format(dir = i))
        #         final_compare.save('./result_seq/{dir}/final_compare.png'.format(dir = i))
        #         pred_list = []
        #         gt_list = []
        # plot

        


    ave_psnr = np.mean(np.concatenate(psnr_list))
    print("Average PSNR: {result}".format(result = ave_psnr))
        
    print("Finish")

        

if __name__ == '__main__':
    main()
        
