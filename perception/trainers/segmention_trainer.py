# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by Resnick Xing on 2018/5/10
Muthu Mookiah modified the functions train_gen, val_gen and visual_patch on 2025/04/14.
"""

import sys
import pandas as pd
from openpyxl import Workbook
import keras
import random, numpy as np, cv2
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback, EarlyStopping, LearningRateScheduler
from tensorflow.python.keras.callbacks import ReduceLROnPlateau

sys.path.append('.../perception/bases')
from trainer_base import TrainerBase

sys.path.append('.../configs/utils')
from utils import genMasks, visualize
from img_utils_rgb import my_PreProc_RGB
from img_utils_gray import img_process, img_processgt

class SegmentionTrainer(TrainerBase):
    def __init__(self, model, data, config):
        super(SegmentionTrainer, self).__init__(model, data, config)
        self.model = model
        self.data = data
        self.config = config
        self.callbacks = []
        self.init_callbacks()



    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=self.config.hdf5_path + self.config.exp_name + '_best_weights.h5',
                verbose=1,
                monitor='val_loss',
                mode='auto',
                save_best_only=True
            )

        )

        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.checkpoint,
                write_images=True,
                write_graph=True,
            )
        )


        self.callbacks.append(
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                mode='min',
                verbose=1,
                min_delta=0.0001
            )

        )


    def train(self):
        gen = DataGenerator(self.data, self.config)
        gen.visual_patch()

        hist = self.model.fit_generator(gen.train_gen(),
                                        epochs=self.config.epochs,
                                        steps_per_epoch=self.config.subsample * self.config.total_train / self.config.batch_size,
                                        verbose=1,
                                        callbacks=self.callbacks,
                                        validation_data=gen.val_gen(),

                                        validation_steps=int(
                                            self.config.subsample * self.config.total_val / self.config.batch_size)
                                        )

        self.model.save_weights(self.config.hdf5_path + self.config.exp_name + '_last_weights.h5', overwrite=True)
        hist_df = pd.DataFrame(hist.history)
        with pd.ExcelWriter('.../experiments/VesselNet/combined/unet.xlsx') as writer:
            hist_df.to_excel(writer)


class DataGenerator():
    """
	load image (Generator)
	"""

    def __init__(self, data, config):

        self.train_img = img_process(data[0])
        self.train_gt = img_processgt(data[1])
        self.val_img = img_process(data[2])
        self.val_gt = img_processgt(data[3])

        self.config = config
        self.num_seg_class = config.seg_num


    def _CenterSampler(self, attnlist, class_weight, Nimgs):

        class_weight = class_weight / np.sum(class_weight)
        p = random.uniform(0, 1)
        psum = 0
        for i in range(class_weight.shape[0]):
            psum = psum + class_weight[i]
            if p < psum:
                label = i
                break
        if label == class_weight.shape[0] - 1:
            i_center = random.randint(0, Nimgs - 1)
            x_center = random.randint(0 + int(self.config.patch_width / 2),
                                      self.config.width - int(self.config.patch_width / 2))

            y_center = random.randint(0 + int(self.config.patch_height / 2),
                                      self.config.height - int(self.config.patch_height / 2))
        else:

            t = attnlist[label]


            cid = random.randint(0, t[0].shape[0] - 1)
            i_center = t[0][cid]
            y_center = t[1][cid] + random.randint(0 - int(self.config.patch_width / 2),
                                                  0 + int(self.config.patch_width / 2))
            x_center = t[2][cid] + random.randint(0 - int(self.config.patch_width / 2),
                                                  0 + int(self.config.patch_width / 2))

        if y_center < self.config.patch_width / 2:
            y_center = self.config.patch_width / 2
        elif y_center > self.config.height - self.config.patch_width / 2:
            y_center = self.config.height - self.config.patch_width / 2

        if x_center < self.config.patch_width / 2:
            x_center = self.config.patch_width / 2
        elif x_center > self.config.width - self.config.patch_width / 2:
            x_center = self.config.width - self.config.patch_width / 2

        return i_center, x_center, y_center



    def _genDef(self, train_imgs, train_masks, attnlist, class_weight):



        while 1:
            Nimgs = train_imgs.shape[0]
            for t in range(int(self.config.subsample * self.config.total_train / self.config.batch_size)):

                X = np.zeros(
                    [self.config.batch_size, self.config.patch_height, self.config.patch_width, 3])  # RGB-3, RG-1
                Y = np.zeros([self.config.batch_size, self.config.patch_height * self.config.patch_width,self.config.seg_num+1])# Original self.config.seg_num+1

                for j in range(self.config.batch_size):
                    [i_center, x_center, y_center] = self._CenterSampler(attnlist, class_weight, Nimgs)
                    patch = train_imgs[i_center,
                            int(y_center - self.config.patch_height / 2):int(y_center + self.config.patch_height / 2),
                            int(x_center - self.config.patch_width / 2):int(x_center + self.config.patch_width / 2), :]


                    patch_mask = train_masks[i_center, :, int(y_center - self.config.patch_height / 2):int(
                        y_center + self.config.patch_height / 2), int(x_center - self.config.patch_width / 2):int(
                        x_center + self.config.patch_width / 2)]

                    X[j, :, :, :] = patch
                    Y[j, :, :] = genMasks(np.reshape(patch_mask, [1, self.config.seg_num, self.config.patch_height,
                                                                  self.config.patch_width]), self.config.seg_num) # Original

                yield (X,Y)



    def train_gen(self):


        class_weight = [1.0, 1.0]
        attnlist_0=[np.where(self.train_gt[:, 0, :, :] == np.max(self.train_gt[:, 0, :, :]))] # Original
        attnlist_1=[np.where(self.train_gt[:, 1, :, :] == np.max(self.train_gt[:, 1, :, :]))] # Original
        attnlist = attnlist_0 + attnlist_1  # MRK AV
        return self._genDef(self.train_img, self.train_gt, attnlist, class_weight)

    def val_gen(self):
        class_weight = [1.0, 1.0]  # [1.0,0.0]AV
        attnlist_0=[np.where(self.val_gt[:,0, :, :] == np.max(self.val_gt[:,0, :, :]))] # Original
        attnlist_1 = [np.where(self.val_gt[:, 1, :, :] == np.max(self.val_gt[:, 1, :, :]))]  # Original
        attnlist = attnlist_0 + attnlist_1  # MRK AV
        return self._genDef(self.val_img, self.val_gt, attnlist, class_weight)

    def visual_patch(self):
        gen = self.train_gen()
        (X, Y) = next(gen)

        image = []
        maska = []
        maskv = []
        masku = []

        print("[INFO] Visualize Image Sample...")
        for index in range(self.config.batch_size):
            image.append(X[index])
            maska.append(np.reshape(Y, [self.config.batch_size, self.config.patch_height, self.config.patch_width,
                                        self.config.seg_num + 1])[index, :, :, 0])
            masku.append(np.reshape(Y, [self.config.batch_size, self.config.patch_height, self.config.patch_width,
                                        self.config.seg_num + 1])[index, :, :, 1])
            maskv.append(np.reshape(Y, [self.config.batch_size, self.config.patch_height, self.config.patch_width,
                                        self.config.seg_num +1])[index, :, :, 2])

        if self.config.batch_size % 4 == 0:
            row = self.config.batch_size / 4
            col = 4
        else:
            if self.config.batch_size % 5 != 0:
                row = self.config.batch_size // 5 + 1
            else:
                row = self.config.batch_size // 5
            col = 5
        imagePatch = visualize(image, [row, col])
        maskPatchA = visualize(maska, [row, col])
        maskPatchU = visualize(masku, [row, col])
        maskPatchV = visualize(maskv, [row, col])

        cv2.imwrite(self.config.checkpoint + "image_patch.jpg", imagePatch)
        cv2.imwrite(self.config.checkpoint + "groundtruth_patch_FP.jpg", maskPatchA) #R
        cv2.imwrite(self.config.checkpoint + "groundtruth_patch_BG.jpg", maskPatchV) #G
        cv2.imwrite(self.config.checkpoint + "groundtruth_patch_BF.jpg", maskPatchU) #B

