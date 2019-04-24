#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 20:11:19 2016

@author: stephen
"""

from __future__ import print_function

import argparse
import time

from keras.models import Model
from keras.utils import np_utils
import numpy as np

import keras
from keras.callbacks import ReduceLROnPlateau
import pandas as pd
from logger import get_log

def readucr(filename):
    data = np.loadtxt(filename, delimiter=',')
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y


flist = [
    'Adiac', 'Beef', 'CBF', 'ChlorineConcentration',
    'CinC_ECG_torso', 'Coffee', 'Cricket_X', 'Cricket_Y', 'Cricket_Z',
    'DiatomSizeReduction', 'ECGFiveDays', 'FaceAll', 'FaceFour', 'FacesUCR', '50words', 'FISH', 'Gun_Point', 'Haptics',
    'InlineSkate', 'ItalyPowerDemand', 'Lighting2', 'Lighting7', 'MALLAT', 'MedicalImages', 'MoteStrain',
    'NonInvasiveFatalECG_Thorax1',
    'NonInvasiveFatalECG_Thorax2', 'OliveOil', 'OSULeaf', 'SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII',
    'StarLightCurves', 'SwedishLeaf', 'Symbols',
    'synthetic_control', 'Trace', 'TwoLeadECG', 'Two_Patterns', 'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y',
    'uWaveGestureLibrary_Z', 'wafer', 'WordsSynonyms', 'yoga'
]
# flist = ['Adiac']

parser = argparse.ArgumentParser()
parser.add_argument('-epoch', type=int, default=5000, help='epoch')
parser.add_argument('-worker_num', type=int, default=1, help='num')
parser.add_argument('-worker_idx', type=int, default=0, help='idx')
parser.add_argument('-data_dir', type=str, help='data')
parser.add_argument('-one_dataset', type=str, help='data')
parser.add_argument('--log_dir', type=str, default='log/', help="log dir")
parser.add_argument('--annotation', type=str, default='empty', help="annotation for distinguishing")

opt = parser.parse_args()

nb_epochs = opt.epoch
data_dir = opt.data_dir
# data_dir = '/Users/kangrong/tsResearch/tols/UCR_TS_Archive_2015/'
worker_num = opt.worker_num
worker_idx = opt.worker_idx

task_num = int(len(flist) // worker_num)
if opt.one_dataset is not None:
    sub_flist = [opt.one_dataset]
else:
    sub_flist = flist[worker_idx * task_num: (worker_idx + 1) * task_num]
    if worker_idx == worker_num - 1:
        sub_flist = flist[worker_idx * task_num:]

logger = get_log(opt.log_dir, "FCN_%s_%d_%d" % (opt.annotation, worker_num, worker_idx))
logger.info(sub_flist)

for each in sub_flist:
    st = time.time()
    logger.info("dataset: %s, start time: %.3f" % (each, st))

    fname = each
    x_train, y_train = readucr(data_dir + fname + '/' + fname + '_TRAIN')
    x_test, y_test = readucr(data_dir + fname + '/' + fname + '_TEST')
    nb_classes = len(np.unique(y_test))
    batch_size = int(min(x_train.shape[0] // 10, 16))

    y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (nb_classes - 1)
    y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (nb_classes - 1)

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    x_train_mean = x_train.mean()
    x_train_std = x_train.std()
    x_train = (x_train - x_train_mean) / (x_train_std)

    x_test = (x_test - x_train_mean) / (x_train_std)
    x_train = x_train.reshape(x_train.shape + (1, 1,))
    x_test = x_test.reshape(x_test.shape + (1, 1,))

    x = keras.layers.Input(x_train.shape[1:])
    #    drop_out = Dropout(0.2)(x)
    conv1 = keras.layers.Conv2D(128, 8, 1, border_mode='same')(x)
    conv1 = keras.layers.normalization.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation('relu')(conv1)

    #    drop_out = Dropout(0.2)(conv1)
    conv2 = keras.layers.Conv2D(256, 5, 1, border_mode='same')(conv1)
    conv2 = keras.layers.normalization.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)

    #    drop_out = Dropout(0.2)(conv2)
    conv3 = keras.layers.Conv2D(128, 3, 1, border_mode='same')(conv2)
    conv3 = keras.layers.normalization.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)

    full = keras.layers.pooling.GlobalAveragePooling2D()(conv3)
    out = keras.layers.Dense(nb_classes, activation='softmax')(full)

    model = Model(input=x, output=out)

    optimizer = keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                                  patience=50, min_lr=0.0001)
    hist = model.fit(x_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs,
                     verbose=1, validation_data=(x_test, Y_test), callbacks=[reduce_lr])
    et = time.time()
    # Print the testing results which has the lowest training loss.
    log = pd.DataFrame(hist.history)
    logger.info("out-db: %s, time: %.4f, min-tr-los: %.4f, max-tr-acc: %.4f, max-val-acc: %.4f" % (
        each, et - st, log.loc[log['loss'].idxmin]['val_acc'],
        log.loc[log['acc'].idxmax]['val_acc'],
        log.loc[log['val_acc'].idxmax]['val_acc']))

