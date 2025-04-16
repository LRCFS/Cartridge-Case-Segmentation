# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by Resnick Xing on 2018/5/10
Muthu Mookiah modified lines 30-52 on 2025/04/14.
"""
import argparse
import json
import sys
import os
sys.path.append('.../configs/utils')
from bunch import Bunch
from utils import mkdir_if_not_exist


def get_config_from_json(json_file):

    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    config = Bunch(config_dict)

    return config, config_dict



def process_config(json_file):

    # Dataset path setting
    config, _ = get_config_from_json(json_file)
    config.val_groundtruth_path = os.path.join('./experiments', config.exp_name,'dataset/validate/gt/')
    config.val_img_path=os.path.join('./experiments', config.exp_name,'dataset/validate/images/')
    config.train_groundtruth_path = os.path.join("./experiments", config.exp_name,"dataset/train/gt/")
    config.train_img_path = os.path.join('./experiments', config.exp_name,'datasettrain/images/')

    config.hdf5_path = os.path.join('./experiments', config.exp_name,'dataset/hdf5/final/unet/')  # original
    config.checkpoint = os.path.join('./experiments', config.exp_name, 'dataset/checkpoint/unet/')
    config.test_img_path=os.path.join('./experiments', config.exp_name, 'dataset/test/images/')# Original
    config.test_gt_path=os.path.join('./experiments', config.exp_name, 'datasettest/gt/')

    config.test_result_path_FP  = os.path.join('./experiments', config.exp_name, 'dataset/results/unet/otsu/FP/')
    config.test_result_path_BF = os.path.join('./experiments', config.exp_name,'dataset/results/unet/otsu/BF/')
    config.test_result_path_BG = os.path.join('./experiments', config.exp_name,'dataset/results/unet/otsu/BG/')

    mkdir_if_not_exist(config.hdf5_path)
    mkdir_if_not_exist(config.checkpoint)
    mkdir_if_not_exist(config.test_img_path)
    mkdir_if_not_exist(config.test_gt_path)

    mkdir_if_not_exist(config.test_result_path_FP)
    mkdir_if_not_exist(config.test_result_path_BF)
    mkdir_if_not_exist(config.test_result_path_BG)
    return config




