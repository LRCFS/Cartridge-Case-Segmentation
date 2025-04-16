# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by Resnick Xing on 2018/5/10
Modified by Muthu R K Mookiah on 2019
"""
import argparse
import json
import sys
import os
sys.path.append('/media/muthu/Data/Projects/PelletAnalysisData/NIST-Dataset/cc/fivefoldbf/fpseg/Codes')
from bunch import Bunch
import numpy as np
from utils import mkdir_if_not_exist


def get_config_from_json(json_file):

    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    config = Bunch(config_dict)

    return config, config_dict



def process_config(json_file):




    ### Combined
    config, _ = get_config_from_json(json_file)
    config.hdf5_path = os.path.join('./experiments', config.exp_name,'hdf5/')  # original
    config.test_img_path=os.path.join('./experiments', config.exp_name, 'test/imgs/')# Original





    config.test_result_path_FP = os.path.join('./experiments', config.exp_name, 'test/results/FP/')
    config.test_result_path_BF = os.path.join('./experiments', config.exp_name,'test/results/BF/')
    config.test_result_path_BG = os.path.join('./experiments', config.exp_name,'test/results/BG/')

    mkdir_if_not_exist(config.hdf5_path)
    mkdir_if_not_exist(config.test_img_path)


    mkdir_if_not_exist(config.test_result_path_FP)
    mkdir_if_not_exist(config.test_result_path_BF)
    mkdir_if_not_exist(config.test_result_path_BG)

    return config




