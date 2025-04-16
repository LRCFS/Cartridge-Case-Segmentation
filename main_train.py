"""
Copyright (c) 2018. All rights reserved.
Created by Resnick Xing on 2018/5/11
Muthu Mookiah modified lines 43, 51, and 53 on 2025/04/14.
"""

import os
import sys
sys.path.append('.../experiments/data_loaders')
from standard_loader import DataLoader
# from infers.simple_mnist_infer import SimpleMnistInfer
sys.path.append('.../perception/models')
from unet import  SegmentionModel
sys.path.append('.../perception/trainers')
from segmention_trainer import SegmentionTrainer
sys.path.append('.../configs/utils')
from config_utils import process_config
import numpy as np
import time


def main_train():

    print('[INFO] Reading Configs...')

    config = None

    try:
        config = process_config('.../configs/segmention_config.json')
    except Exception as e:
        print('[Exception] Config Error, %s' % e)
        exit(0)

    print('[INFO] Preparing Data...')
    dataloader = DataLoader(config=config)
    dataloader.prepare_dataset()

    train_imgs,train_gt=dataloader.get_train_data()
    val_imgs,val_gt=dataloader.get_val_data()

    print('[INFO] Building Model...')
    model = SegmentionModel(config=config)
    st = time.time()
    print('[INFO] Training...')
    trainer = SegmentionTrainer(
         model=model.model,
         data=[train_imgs,train_gt,val_imgs,val_gt],
         config=config)
    trainer.train()
    print('[INFO] Finishing...')
    et = time.time()
    # get the execution time
    elapsed_time = et-st
    print ('Execution time', elapsed_time, 'seconds')
if __name__ == '__main__':
    main_train()

