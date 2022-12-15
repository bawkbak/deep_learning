import torch
import os
import numpy as np
import csv
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


default_transform = transforms.Compose([
    transforms.ToTensor(),
    ])


class bair_robot_pushing_dataset(Dataset):
    def __init__(self, args, mode='train', transform=default_transform):
        self.mode = mode
        self.data_folder = os.path.join('./' + self.mode)
        self.sequence_folder = []
        self.seq_len = args.n_past + args.n_future # ([seq_len, 3, 64, 64])
        self.seed_is_set = False
        self.d = 0
        self.d_now = None
        if self.mode == 'train':
            self.ordered = False
        else:
            self.ordered = True
        self.ordered = False
        for d1 in os.listdir(self.data_folder):
            # print(d1)
            for d2 in os.listdir('%s/%s' % (self.data_folder, d1)):
                # print(d2)
                self.sequence_folder.append('%s/%s/%s' % (self.data_folder, d1, d2))
        # print("###############")
        # print(self.sequence_folder)
        print("> Found %d sequence..." % (len(self.sequence_folder)))
    
    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
        

    def __len__(self):
        return len(self.sequence_folder) 
        
    def get_seq(self):
        if self.ordered:
            d = self.sequence_folder[self.d]
            if self.d == len(self.sequence_folder) - 1:
                self.d = 0
            else:
                self.d+=1
        else:
            d = self.sequence_folder[np.random.randint(len(self.sequence_folder))]
        image_seq = []
        self.d_now = d
        for i in range(self.seq_len):
            fname = '%s/%d.png' % (d, i)
            img = Image.open(fname) 
            img = np.array(img).reshape(1, 64, 64, 3)
            image_seq.append(img/255.)
        
        image_seq = np.concatenate(image_seq, axis=0)
        image_seq = torch.from_numpy(image_seq)
        image_seq = image_seq.permute(0,3,1,2)
        # image_seq = np.transpose(image_seq, (0,3,1,2)).shape
        # print("random: ", d)

        # a = Image.open(tmp_x) 
        # a.save('now:aaa.png')

        # tt = image_seq.numpy()
        # # print("type: ", type(tt))
        # # print("shape: ", np.shape(tt[1,:,:,:]))
        # # data = np.array(Image.fromarray((tt[1,:,:,:] * 255).astype(np.uint8)).resize((64, 64)).convert('RGB'))
        # tt = (np.transpose(tt[1,:,:,:], (1, 2, 0)) + 1) / 2.0 * 555.0
        # data = Image.fromarray(np.uint8(tt))
        # data.save('now:ttt.png')

        return image_seq




    def get_csv(self):
        action = None 
        position = None 
        data = None

        with open('%s/actions.csv'% (self.d_now), newline='') as csvfile:
            data = list(csv.reader(csvfile))
        action = np.array(data)


        with open('%s/endeffector_positions.csv'% (self.d_now), newline='') as csvfile:
            data = list(csv.reader(csvfile))
        position = np.array(data)


        c = np.concatenate((action,position),axis=1)
        c = c[:self.seq_len]
        c = c.astype(np.float)

        return c
    
    def __getitem__(self, index):
        self.set_seed(index)
        seq = self.get_seq()
        cond =  self.get_csv()
        
        return seq, cond