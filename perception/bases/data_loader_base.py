
"""
Copyright (c) 2018. All rights reserved.
Created by Resnick Xing on 2018/5/10
"""


class DataLoaderBase(object):


    def __init__(self, config):
        self.config = config

    def prepare_dataset(self):

        raise NotImplementedError

    def get_train_data(self):

        raise NotImplementedError

    def get_val_data(self):

        raise NotImplementedError
