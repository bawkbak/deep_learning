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
    parser.add_argument('--model_dir', default='', help='base directory to save logs')
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
    # print("type x: ", type(x))
    x = x.float()
    # cond = cond.to("cuda")

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
            x_in = modules['decoder']([h, skip]).detach()
            gen_seq.append(x_in.data.cpu().numpy())
    gen_seq = torch.tensor(np.array(gen_seq))
    gen_seq = gen_seq.permute(1,0,2,3,4)
    return gen_seq



def train(x, cond, modules, optimizer, kl_anneal, args, iterations):
    
    modules['frame_predictor'].zero_grad()
    modules['posterior'].zero_grad()
    modules['encoder'].zero_grad()
    modules['decoder'].zero_grad()

    # initialize the hidden state.
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    modules['posterior'].hidden = modules['posterior'].init_hidden()
    
    mse = 0
    kld = 0

    use_teacher_forcing = True if random.random() < args.tfr else False

    x = x.float()
    # cond = cond.to('cuda')

    #
    train_result = []
    origin_result = []
    #
    h_seq = [ modules['encoder'](x[:,i]) for i in range(args.n_past + args.n_future)]

    for i in range(1, args.n_past + args.n_future):

        h_target = h_seq[i][0]

        c = cond[:, i, :].float()
        
        if args.last_frame_skip or i < args.n_past:	
            h = h_seq[i-1][0]
            skip = h_seq[i-1][1]
        else:
            h = h_seq[i-1][0]
        
        if i > 1:
            previous_img = x_pred
            pr_latent = modules['encoder'](previous_img)
            h_no_teacher = pr_latent[0]
            # h_no_teacher = pr_latent
        else:
            h_no_teacher = h

        z_t, mu, logvar = modules['posterior'](h_target)

        if use_teacher_forcing:
            h_pred = modules['frame_predictor'](torch.cat([h, z_t, c], 1))
        else:
            # print("without teacher")
            h_pred = modules['frame_predictor'](torch.cat([h_no_teacher, z_t, c], 1))
            
        x_pred = modules['decoder']([h_pred, skip])
        mse += nn.MSELoss()(x_pred, x[:,i])

        kld += kl_criterion(mu, logvar, args)
        
        #111
        # train_result.append(x_pred.data.cpu().numpy())
        # origin_result.append(x[:,i].data.cpu().numpy())
        #111

    
    beta = kl_anneal.get_beta(iterations)
    args.visualize_beta = beta
    loss = mse + kld * beta
    loss.backward()
    #
    # train_result = np.array(train_result)
    # tt = (np.transpose(train_result[1,1,:,:], (1, 2, 0)) + 1) * 255.0
    # data = Image.fromarray(np.uint8(tt))
    # data.save('./cvae_psnr_gen/debug/pic:{now_epoch}_1.png'.format(now_epoch = iterations))

    # origin_result = np.array(origin_result)
    # tt = (np.transpose(origin_result[1,1,:,:], (1, 2, 0)) + 1) * 255.0
    # data = Image.fromarray(np.uint8(tt))
    # data.save('./cvae_psnr_gen/debug/pic:{now_epoch}_2.png'.format(now_epoch = iterations))
    #

    optimizer.step()
    length_batch = args.n_past + args.n_future
    return loss.detach().cpu().numpy() / (length_batch), mse.detach().cpu().numpy() / (length_batch), kld.detach().cpu().numpy() / (length_batch)

class kl_annealing():
    def __init__(self, args):
        super().__init__()
        # raise NotImplementedError
        self.niter = args.niter
        self.is_cycle = args.kl_anneal_cyclical
        self.kl_anneal_ratio = args.kl_anneal_ratio
        self.kl_anneal_cycle = args.kl_anneal_cycle
        self.period = (self.niter // self.kl_anneal_cycle) 
        self.monotonic_ratio = 1 / (args.kl_anneal_ratio * args.niter)
        self.cyclical_ratio = 1 / self.period  / self.kl_anneal_ratio

        
    def get_beta(self, iterations):
        if self.is_cycle == False: #monotonic
            KL_weight = iterations * self.monotonic_ratio
            return KL_weight if KL_weight < 1 else 1.
        else: #cyclical
            if iterations % self.period <= self.kl_anneal_ratio * self.period:
                KL_weight = iterations % self.period  * self.cyclical_ratio
                return KL_weight if KL_weight < 1 else 1.
            else:
                return 1
        

           
            

def main():
    #plot array
    KLD_list = []
    beta_list = []
    mse_list = []
    loss_list = []
    PSNR_list = []
    TFR_list = []
    #plot array
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

    if args.model_dir != '':
        # load model and continue training from checkpoint
        saved_model = torch.load('%s/model.pth' % args.model_dir)
        optimizer = args.optimizer
        model_dir = args.model_dir
        niter = args.niter
        args = saved_model['args']
        args.optimizer = optimizer
        args.model_dir = model_dir
        args.log_dir = '%s/continued' % args.log_dir
        start_epoch = saved_model['last_epoch']
    else:
        name = 'rnn_size=%d-predictor-posterior-rnn_layers=%d-%d-n_past=%d-n_future=%d-lr=%.4f-g_dim=%d-z_dim=%d-last_frame_skip=%s-beta=%.7f'\
            % (args.rnn_size, args.predictor_rnn_layers, args.posterior_rnn_layers, args.n_past, args.n_future, args.lr, args.g_dim, args.z_dim, args.last_frame_skip, args.beta)

        args.log_dir = '%s/%s' % (args.log_dir, name)
        niter = args.niter
        start_epoch = 0

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs('%s/gen/' % args.log_dir, exist_ok=True)

    print("Device: ", device)
    print("Random Seed: ", args.seed)
    if args.kl_anneal_cyclical:
        print("KL method: cyclical")
    else:
        print("KL method: monotonic")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if os.path.exists('./{}/train_record.txt'.format(args.log_dir)):
        os.remove('./{}/train_record.txt'.format(args.log_dir))
    
    print(args)

    with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
        train_record.write('args: {}\n'.format(args))

    # ------------ build the models  --------------

    if args.model_dir != '':
        frame_predictor = saved_model['frame_predictor']
        posterior = saved_model['posterior']
    else:
        frame_predictor = lstm(args.g_dim+args.z_dim+args.cond_dim, args.g_dim, args.rnn_size, args.predictor_rnn_layers, args.batch_size, device)
        posterior = gaussian_lstm(args.g_dim, args.z_dim, args.rnn_size, args.posterior_rnn_layers, args.batch_size, device)
        frame_predictor.apply(init_weights)
        posterior.apply(init_weights)
            
    if args.model_dir != '':
        decoder = saved_model['decoder']
        encoder = saved_model['encoder']
    else:
        encoder = vgg_encoder(args.g_dim)
        decoder = vgg_decoder(args.g_dim)
        encoder.apply(init_weights)
        decoder.apply(init_weights)
    
    # --------- transfer to device ------------------------------------
    frame_predictor.to(device)
    posterior.to(device)
    encoder.to(device)
    decoder.to(device)

    # --------- load a dataset ------------------------------------
    print("Train data:")
    train_data = bair_robot_pushing_dataset(args, 'train')
    print("validate data:")
    validate_data = bair_robot_pushing_dataset(args, 'validate')
    train_loader = DataLoader(train_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)
    train_iterator = iter(train_loader)

    validate_loader = DataLoader(validate_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)

    validate_iterator = iter(validate_loader)

    # ---------------- optimizers ----------------
    if args.optimizer == 'adam':
        args.optimizer = optim.Adam
    elif args.optimizer == 'rmsprop':
        args.optimizer = optim.RMSprop
    elif args.optimizer == 'sgd':
        args.optimizer = optim.SGD
    else:
        raise ValueError('Unknown optimizer: %s' % args.optimizer)

    params = list(frame_predictor.parameters()) + list(posterior.parameters()) + list(encoder.parameters()) + list(decoder.parameters())
    optimizer = args.optimizer(params, lr=args.lr, betas=(args.beta1, 0.999))
    kl_anneal = kl_annealing(args)

    modules = {
        'frame_predictor': frame_predictor,
        'posterior': posterior,
        'encoder': encoder,
        'decoder': decoder,
    }
    # --------- training loop ------------------------------------

    progress = tqdm(total=args.niter)
    best_val_psnr = 0
    for epoch in range(start_epoch, start_epoch + niter):
        
        frame_predictor.train()
        posterior.train()
        encoder.train()
        decoder.train()

        epoch_loss = 0
        epoch_mse = 0
        epoch_kld = 0

        for _ in range(args.epoch_size):
           
            try:
                seq, cond = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                seq, cond = next(train_iterator)
            # print("range ", _)
            seq = seq.to(device)
            cond = cond.to(device)
            loss, mse, kld = train(seq, cond, modules, optimizer, kl_anneal, args, epoch)
            epoch_loss += loss
            epoch_mse += mse
            epoch_kld += kld
        ### append
        KLD_list.append(epoch_kld/args.epoch_size)
        beta_list.append(args.visualize_beta)
        mse_list.append(epoch_mse/args.epoch_size)
        loss_list.append(epoch_loss/args.epoch_size)
        TFR_list.append(args.tfr)
        ### append
        if epoch >= args.tfr_start_decay_epoch:
            args.tfr = 1. - (1 - args.tfr_lower_bound) / (args.niter - args.tfr_start_decay_epoch) * (epoch - args.tfr_start_decay_epoch) 
        if args.tfr < args.tfr_lower_bound: 
            args.tfr = args.tfr_lower_bound

        # print('now:{now_epoch}/total:{arg_epoch} loss: {loss}, mse loss: {mse_loss}, kld loss: {kld_loss}'.format(
        #     now_epoch = epoch, arg_epoch = args.niter, loss = epoch_loss/args.epoch_size, mse_loss = epoch_mse/args.epoch_size, kld_loss = epoch_kld/args.epoch_size))
        progress.update(1)
        with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
            train_record.write(('[epoch: %02d] loss: %.5f | mse loss: %.5f | kld loss: %.5f\n' % (epoch, epoch_loss/args.epoch_size, epoch_mse/args.epoch_size, epoch_kld/args.epoch_size)))
            # torch.save({
            # 'encoder': encoder,
            # 'decoder': decoder,
            # 'frame_predictor': frame_predictor,
            # 'posterior': posterior,
            # 'args': args},
            # '%s/model.pth' % args.log_dir)
        

        ##################### EVALUATE #########################

        frame_predictor.eval()
        encoder.eval()
        decoder.eval()
        posterior.eval()

        if epoch % 5 == 0:
            psnr_list = []
            for _ in range(len(validate_data) // args.batch_size):
                try:
                    validate_seq, validate_cond = next(validate_iterator)
                except StopIteration:
                    validate_iterator = iter(validate_loader)
                    validate_seq, validate_cond = next(validate_iterator)
                
                validate_seq = validate_seq.to(device)
                validate_cond = validate_cond.to(device)
                pred_seq = pred(validate_seq, validate_cond, modules, args, device)
                _, _, psnr = finn_eval_seq(validate_seq[:, args.n_past:].cpu(), pred_seq.cpu().numpy())
                psnr_list.append(psnr)

                #111
                pred_seq = np.array(pred_seq)
                tt = (np.transpose(pred_seq[1,1,:,:], (1, 2, 0)) + 1) * 255.0
                data = Image.fromarray(np.uint8(tt))
                data.save('./cvae_psnr_gen/psnr:{now_epoch}.png'.format(now_epoch = epoch))
                #111


            ave_psnr = np.mean(np.concatenate(psnr_list))
            print("PSNR: ", ave_psnr)
            PSNR_list.append(ave_psnr)

            with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
                train_record.write(('====================== validate psnr = {:.5f} ========================\n'.format(ave_psnr)))

            if ave_psnr > best_val_psnr:
                best_val_psnr = ave_psnr
                # save the model
                print("save pth: ", best_val_psnr)
                torch.save({
                    'encoder': encoder,
                    'decoder': decoder,
                    'frame_predictor': frame_predictor,
                    'posterior': posterior,
                    'args': args,
                    'last_epoch': epoch},
                    '%s/final_model.pth' % args.log_dir)

        if epoch == 299: 
            plot_result(KLD_list, mse_list, loss_list, PSNR_list, beta_list, TFR_list, epoch+1, args)
            plot_psnr(PSNR_list, epoch+1)
            with open('./plot_record.txt', 'w') as result:
                result.write('kld: {}\n'.format(KLD_list))
                result.write('\nmse: {}\n'.format(mse_list))
                result.write('\nloss: {}\n'.format(loss_list))
                result.write('\npsnr: {}\n'.format(PSNR_list))
                result.write('\nbeta: {}\n'.format(beta_list))
                result.write('\ntfr: {}\n'.format(TFR_list))
        if epoch % 20 == 0:
            plot_result(KLD_list, mse_list, loss_list, PSNR_list, beta_list, TFR_list, epoch+1, args)
            plot_psnr(PSNR_list, epoch+1)
            with open('./plot_record.txt', 'w') as result:
                result.write('kld: {}\n'.format(KLD_list))
                result.write('\nmse: {}\n'.format(mse_list))
                result.write('\nloss: {}\n'.format(loss_list))
                result.write('\npsnr: {}\n'.format(PSNR_list))
                result.write('\nbeta: {}\n'.format(beta_list))
                result.write('\ntfr: {}\n'.format(TFR_list))

            # try:
            #     validate_seq, validate_cond = next(validate_iterator)
            # except StopIteration:
            #     validate_iterator = iter(validate_loader)
            #     validate_seq, validate_cond = next(validate_iterator)
            # validate_seq = validate_seq.to(device)
            # plot_psnr(validate_seq, validate_cond, modules, epoch, args)
            # plot_kl(validate_seq, validate_cond, modules, epoch, args)
        

if __name__ == '__main__':
    main()
        
