"""
Copyright (c) 2018. All rights reserved.
Created by Resnick Xing on 2018/5/10
"""


class TrainerBase(object):


    def __init__(self, model, data, config):
        self.model = model
        self.data = data
        self.config = config

    def train(self):

        raise NotImplementedError
