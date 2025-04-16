
"""
Copyright (c) 2018. All rights reserved.
Created by Resnick Xing on 2018/5/10
"""


class InferBase(object):


    def __init__(self, config):
        self.config = config

    def load_model(self, name):

        raise NotImplementedError

    def predict(self, data):

        raise NotImplementedError
