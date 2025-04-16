# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by Resnick Xing on 2018/5/10
Modified by Muthu R K Mookiah on 2019
"""


import os
import glob,cv2,numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
import sys
sys.path.append('/media/muthu/Data/Projects/PelletAnalysisData/NIST-Dataset/cc/fivefoldbf/fpseg/Codes')
from infer_base import InferBase
from img_utils_gray import get_test_patches,pred_to_patches,recompone_overlap
from img_utils_gray import img_process, imgResize,imgResizeOri,rgb2gray


class SegmentionInfer(InferBase):
	def __init__(self,config):
		super(SegmentionInfer, self).__init__(config)
		self.load_model()

	def load_model(self):
		self.model = model_from_json(open(self.config.hdf5_path+self.config.exp_name + '_architecture.json').read())# Orignal
		self.model.load_weights(self.config.hdf5_path+self.config.exp_name+ '_best_weights.h5')# Original

	def analyze_name(self,path):
		return (path.split('/')[-1]).split(".")[0]

	def predict(self):
		predList=glob.glob(self.config.test_img_path+"*."+self.config.test_datatype)

		for path in predList:
			print('test img path',path)

			orgImg_temp1 = cv2.imread(path)
			orgImg = imgResize(orgImg_temp1)

			height,width=orgImg.shape[:2]
			orgImg = np.reshape(orgImg, (height,width,3))

			patches_pred,new_height,new_width,adjustImg=get_test_patches(orgImg,self.config)
			predictions = self.model.predict(patches_pred, batch_size=1, verbose=1)
			print(predictions.shape)
			pred_patches=pred_to_patches(predictions,self.config)

			pred_imgs=recompone_overlap(pred_patches,self.config,new_height,new_width)
			pred_imgs=pred_imgs[:,0:height,0:width,:]




			probResult0 = pred_imgs[0, :, :, 0]
			probResult1 = pred_imgs[0, :, :, 1]
			probResult2 = pred_imgs[0, :, :, 2]


			probResult0 = imgResizeOri(probResult0, dim=(int(orgImg_temp1.shape[1]), int(orgImg_temp1.shape[0])))
			probResult1 = imgResizeOri(probResult1, dim=(int(orgImg_temp1.shape[1]), int(orgImg_temp1.shape[0])))
			probResult2 = imgResizeOri(probResult2, dim=(int(orgImg_temp1.shape[1]), int(orgImg_temp1.shape[0])))


			binaryResulttemp0 = ((probResult0 * 255).astype(np.uint8))
			binaryResulttemp1 = ((probResult1 * 255).astype(np.uint8))
			binaryResulttemp2 = ((probResult2 * 255).astype(np.uint8))



			_,binaryResultb0= cv2.threshold(binaryResulttemp0, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
			_,binaryResultb1= cv2.threshold(binaryResulttemp1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
			_,binaryResultb2= cv2.threshold(binaryResulttemp2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)



			cv2.imwrite(self.config.test_result_path_FP + self.analyze_name(path)+ "_FP.png",binaryResultb0.astype(np.uint8))
			cv2.imwrite(self.config.test_result_path_BF + self.analyze_name(path) + "_BF.png",binaryResultb1.astype(np.uint8))
			cv2.imwrite(self.config.test_result_path_BG + self.analyze_name(path) + "_BG.png",binaryResultb2.astype(np.uint8))




