
"""
Copyright (c) 2018. All rights reserved.
Created by Resnick Xing on 2018/5/11
Modified by Muthu R K Mookiah on 2019
"""
import sys
sys.path.append('/media/muthu/Data/Projects/PelletAnalysisData/NIST-Dataset/cc/fivefoldbf/fpseg/Codes')
from segmention_infer import SegmentionInfer
from config_utils import process_config


repredict=True

def main_test():
    print('[INFO] Reading Configs...')
    config = None

    try:
        config = process_config('segmention_config.json')
    except Exception as e:
        print('[Exception] Config Error, %s' % e)
        exit(0)

    if repredict==True:

        print('[INFO] Predicting...')
        infer = SegmentionInfer(config)
        infer.predict()

if __name__ == '__main__':
    main_test()
