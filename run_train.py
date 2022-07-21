import pandas as pd 
import numpy as np
import glob
import cv2 as cv
import os

import json
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
from torchvision.models import resnet18

import imgaug.augmenters as iaa
import matplotlib.pyplot as plt

from IPython.display import clear_output
from PIL import Image

from core.dataset import ImageDataset
from core.opts import trainer_opts, optimization_opts, dataset_opts
from core.model import BaselineModel, CnnAttentionMixedModel
from core.utils import calc_metric, Cutout

import wandb


def get_checkpoint(model, optimizer, epoch, scheduler=None):
    ckpt = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        # 'scheduler': scheduler.state_dict(),
        'epoch': epoch
    }
    return ckpt



train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224),
    Cutout(64, 0.4),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

orig_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class Trainer:
    def __init__(self, trainer_opts, optimization_opts, dataset_opts):
        self.trainer_opts = trainer_opts
        self.opt_opts = optimization_opts
        self.data_opts = dataset_opts
        
        self.epoch_num = trainer_opts.epoch_num
        self.device = trainer_opts.device
        self._preprocess_original_img()
        self.setup()
        
    def setup(self):
        self.setup_trainers()
        self.setup_opt()
        wandb.init(
            project='MIPT_champ',
            name='launch',
        )
    
    def _preprocess_original_img(self):
        orig_s = self.data_opts.original_size
        scale = self.data_opts.scale
        
        self.original = np.array(Image.open('original.tiff').resize((orig_s // scale, orig_s // scale)))
        self.orig_img = orig_transform(self.original).unsqueeze(0).float().to(self.device)
    
    def setup_opt(self):
        self.model = BaselineModel(out_ch=5, pretrained=False)
        
        if hasattr(self.trainer_opts, 'ckpt_start'):
            print('Downloading weights')
            ckpt = torch.load(self.trainer_opts.ckpt_start)
            self.model.load_state_dict(ckpt)
        
        self.model.to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.opt_opts.lr,
            betas=self.opt_opts.betas, 
            weight_decay=self.opt_opts.weight_decay
        )
        self.criterion = torch.nn.MSELoss(reduction='mean')
        # self.shcheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     self.optimizer, max_lr=self.opt_opts.max_lr, steps_per_epoch=len(self.train_loader), epochs=self.epoch_num
        # )
        
        
    def setup_trainers(self):
        train_root = 'train'
        valid_root = 'eval'
        
        train_dataset = ImageDataset(train_root, self.original, self.data_opts, train_transform)
        valid_dataset = ImageDataset(
            valid_root, self.original, self.data_opts, valid_transform, training=False
        )

        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.opt_opts.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

        self.valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset,
            batch_size=self.opt_opts.batch_size,
            num_workers=4,
            pin_memory=True,
        )
    
    def train(self):
        train_loss_log = []
        
        best_metric = 0.
        best_ckpt = None
        
        for epoch in tqdm(range(self.epoch_num)):
            self.model.train()
            epoch_loss = []

            for imgs, labels in self.train_loader:
                imgs = imgs.float().to(self.device)
                labels = labels.float().to(self.device)
                y_pred = self.model(self.orig_img, imgs)
                
                self.optimizer.zero_grad()
                loss = self.criterion(y_pred, labels)
                loss.backward()
                epoch_loss.append(loss.item())
                # print(epoch_loss[-1])
                self.optimizer.step()
                # self.shcheduler.step()

            if (epoch + 1) % 15 == 0:
                torch.save(self.model.state_dict(), f'models/model_{epoch}.pt')

            if (epoch + 1) % 50 == 0:
                ckpt = get_checkpoint(self.model, self.optimizer, epoch, None)
                torch.save(ckpt, f'ckpt/ckpt_{epoch}.pt')
            
            if (epoch + 1) % 20 == 0:
                eval_metric = self.eval_model()
                wandb.log({
                    'eval_metric': eval_metric
                })
                if eval_metric > best_metric:
                    best_ckpt = self.model.state_dict()
                    best_metric = eval_metric
                    torch.save(best_ckpt, f'models/best_model.pt')
                    
            train_loss_log.append(np.mean(epoch_loss))
            wandb.log({"train_loss": train_loss_log[-1]})
        
        return train_loss_log
    
    @torch.no_grad()
    def eval_model(self):
        self.model.eval()
        
        answer = 0.
        denom = 0.
        
        for imgs, labels in tqdm(self.valid_loader):
            imgs = imgs.float().to(self.device)
            labels = labels.float().to(self.device)
            y_pred = self.model(self.orig_img, imgs)
            
            answer += calc_metric(y_pred, labels)
            denom += imgs.size(0)
        
        return answer / denom
        

if __name__ == '__main__':
    trainer = Trainer(
        trainer_opts,
        optimization_opts,
        dataset_opts
    )
    
    trainer.train()
    