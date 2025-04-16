"""
Copyright (c) 2018. All rights reserved.
Created by Resnick Xing on 2018/5/10
Muthu Mookiah modified the function i_access_dataset on 2025/04/14.
"""
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('.../perception/bases')
from data_loader_base import DataLoaderBase
sys.path.append('.../configs/utils')
from utils import write_hdf5,load_hdf5
import os
import cv2


class DataLoader(DataLoaderBase):
	def __init__(self, config=None):
		super(DataLoader, self).__init__(config)

		self.train_img_path=config.train_img_path

		self.train_groundtruth_path = config.train_groundtruth_path
		self.train_type=config.train_datatype
		self.val_img_path=config.val_img_path
		self.val_groundtruth_path=config.val_groundtruth_path
		self.val_type = config.val_datatype


		self.exp_name=config.exp_name
		self.hdf5_path=config.hdf5_path
		self.height=config.height
		self.width=config.width
		self.num_seg_class=config.seg_num


	def _access_dataset(self,origin_path,groundtruth_path,datatype):

		orgList = glob.glob(origin_path+"*."+datatype)
		gtList = glob.glob(groundtruth_path + "*.png")
		print('gt list name',gtList)

		for num in range(len(orgList)):

			loc1=orgList[num].rfind('/')
			loc2 = orgList[num].rfind('.')
			gtList[num] = groundtruth_path + orgList[num][loc1+1:loc2] + '.png'  # AV TOM

		assert (len(orgList) == len(gtList))

		if os.path.exists('imgsa.npy'):
			imgs = np.memmap('imgsa11.npy', dtype='float32', mode='w+', shape=(len(orgList), self.height, self.width, 3))
		else:
			imgs = np.memmap('imgsa.npy', dtype='float32', mode='w+', shape=(len(orgList), self.height, self.width, 3))

		if os.path.exists('Gt.npy'):
			groundTruth = np.memmap('Gt11.npy', dtype='float32', mode='w+', shape=(len(gtList), self.num_seg_class, self.height, self.width))
		else:
			groundTruth = np.memmap('Gt.npy', dtype='float32', mode='w+', shape=(len(gtList), self.num_seg_class, self.height, self.width))

		for index in range(len(orgList)):
			orgPath=orgList[index]
			orgImg = cv2.imread(orgPath)
			imgs[index] = np.asarray(orgImg)

			for no_seg in range(self.num_seg_class):
				gtPath=gtList[index]
				gtImg = cv2.imread(gtPath)
				groundTruth[index,no_seg] = np.asarray(gtImg[:,:,no_seg])

		print("[INFO] Reading...")
		assert (np.max(groundTruth) == 255)
		assert (np.min(groundTruth) == 0)
		return imgs,groundTruth


	def prepare_dataset(self):


		self.imgs_train, self.groundTruth=self._access_dataset(self.train_img_path,self.train_groundtruth_path,self.train_type)
		write_hdf5(self.imgs_train,self.hdf5_path+"train_img.hdf5")
		write_hdf5(self.groundTruth, self.hdf5_path+"train_groundtruth.hdf5")
		print("[INFO] Saving Training Data")

		self.imgs_val, self.groundTruth_val = self._access_dataset(self.val_img_path, self.val_groundtruth_path, self.val_type)
		write_hdf5(self.imgs_val, self.hdf5_path + "val_img.hdf5")
		write_hdf5(self.groundTruth_val, self.hdf5_path + "val_groundtruth.hdf5")
		print("[INFO] Saving Validation Data")

	def get_train_data(self):

		imgs_train = load_hdf5(self.hdf5_path+"train_img.hdf5")
		groundTruth = load_hdf5(self.hdf5_path+"train_groundtruth.hdf5")
		return imgs_train,groundTruth
		print("[INFO] Loading Training Data")

	def get_val_data(self):
		imgs_val = load_hdf5(self.hdf5_path+"val_img.hdf5")
		groundTruth_val = load_hdf5(self.hdf5_path+"val_groundtruth.hdf5")
		return imgs_val,groundTruth_val
		print("[INFO] Loading Validation Data")