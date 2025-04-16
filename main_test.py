"""
Copyright (c) 2018. All rights reserved.
Created by Resnick Xing on 2018/5/11
"""
import sys
sys.path.append('.../perception/infers')
from segmention_infer import SegmentionInfer
sys.path.append('.../perception/metric')
from segmention_metric import *
sys.path.append('.../configs/utils')
from config_utils import process_config


repredict=True

def main_test():
    print('[INFO] Reading Configs...')
    config = None

    try:
        config = process_config('configs/segmention_config.json')
    except Exception as e:
        print('[Exception] Config Error, %s' % e)
        exit(0)

    if repredict==True:

        print('[INFO] Predicting...')
        infer = SegmentionInfer(config)
        infer.predict()

if __name__ == '__main__':
    main_test()
