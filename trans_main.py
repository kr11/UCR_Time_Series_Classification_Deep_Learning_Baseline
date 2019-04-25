#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 21:46:23 2016

@author: stephen
"""

from __future__ import print_function
import tensorflow as tf
import argparse
import time
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.utils import np_utils
import numpy as np
import keras
from keras.callbacks import ReduceLROnPlateau
import pandas as pd
from logger import get_log
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.initializers import *

# np.random.seed(813306)
from transformer import EncoderLayer, LRSchedulerPerStep, LayerNormalization, PositionwiseFeedForward


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
parser.add_argument('-log_dir', type=str, default='log/', help="log dir")
parser.add_argument('-annotation', type=str, default='empty', help="annotation for distinguishing")

parser.add_argument('-d_inner_hid', type=int, default=1, help='num')
parser.add_argument('-n_head', type=int, default=1, help='num')
parser.add_argument('-d_k', type=int, default=1, help='num')
parser.add_argument('-d_v', type=int, default=1, help='num')
parser.add_argument('-layers', type=int, default=1, help='num')

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

logger = get_log(opt.log_dir, "Trans_%s_%d_%d" % (opt.annotation, worker_num, worker_idx))

logger.info(sub_flist)
print(sub_flist)

for each in sub_flist:
    st = time.time()
    print("dataset: %s, start time: %.3f" % (each, st))
    logger.info("dataset: %s, start time: %.3f" % (each, st))
    fname = each
    x_train, y_train = readucr(data_dir + fname + '/' + fname + '_TRAIN')
    x_test, y_test = readucr(data_dir + fname + '/' + fname + '_TEST')
    nb_classes = len(np.unique(y_test))
    y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (nb_classes - 1)
    y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (nb_classes - 1)
    batch_size = int(min(x_train.shape[0] / 10, 16))

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    x_train_mean = x_train.mean()
    x_train_std = x_train.std()
    x_train = (x_train - x_train_mean) / (x_train_std)

    # x_test_min = np.min(x_test, axis = 1, keepdims=1)
    # x_test_max = np.max(x_test, axis = 1, keepdims=1)
    x_test = (x_test - x_train_mean) / (x_train_std)


    # x_train = x_train.reshape(x_train.shape + (1,))
    # x_test = x_test.reshape(x_test.shape + (1,))
    # def expand_dim_backend(x):
    #     return K.expand_dims(x, -1)
    #
    #
    def squeeze_dim_backend(y):
        # return K.squeeze(y, -1)
        return K.sum(y, 2)


    # x_train = Lambda(expand_dim_backend)(x_train)
    # x_test = Lambda(expand_dim_backend)(x_test)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    x = Input(shape=x_train.shape[1:])
    # y = Dropout(0.1)(x)
    # y = Dense(500, activation='relu')(x)
    # y = Dropout(0.2)(y)
    # y = Dense(500, activation='relu')(y)
    # y = Dropout(0.2)(y)
    # y = Dense(500, activation='relu')(y)
    # y = Dropout(0.3)(y)
    # d_model = 1
    d_inner_hid = opt.d_inner_hid #1  # d_inner_hid = 512
    n_head = opt.n_head # 1  # n_head = 3
    d_k = opt.d_k # 1 #64
    d_v = opt.d_v # 1 #64
    layers = opt.layers # 1
    dropout_rate = 0.1
    encodeLayerList = [EncoderLayer(1, d_inner_hid, n_head, d_k, d_v, dropout_rate) for _ in range(layers)]
    y = None
    for enc_layer in encodeLayerList:
        if y is None:
            y, _ = enc_layer(x)
        else:
            y, _ = enc_layer(y)

    y_2dim = Reshape([int(y.shape[1])])(y)
    # y_2dim = Reshape([int(x.shape[1])])(x)

    out = Dense(nb_classes, activation='softmax')(y_2dim)

    model = Model(input=x, output=out)

    # Res
    optimizer = keras.optimizers.Adam()
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                                  patience=50, min_lr=0.0001)
    # transformer
    # reduce_lr = LRSchedulerPerStep(d_model, 4000)  # there is a warning that it is slow, however, it's ok.
    # optimizer = keras.optimizers.Adam(0.0001, 0.9, 0.98, epsilon=1e-9)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    hist = model.fit(x_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs,
                     verbose=1, validation_data=(x_test, Y_test),
                     # callbacks = [TestCallback((x_train, Y_train)), reduce_lr, keras.callbacks.TensorBoard(log_dir='./log'+fname, histogram_freq=1)])
                     callbacks=[reduce_lr])
    et = time.time()

    # Print the testing results which has the lowest training loss.
    log = pd.DataFrame(hist.history)
    log_info = "out-db: %s, time: %.4f, min-tr-los: %.4f, max-tr-acc: %.4f, max-val-acc: %.4f" % (
        each, et - st, log.loc[log['loss'].idxmin]['val_acc'],
        log.loc[log['acc'].idxmax]['val_acc'],
        log.loc[log['val_acc'].idxmax]['val_acc'])
    logger.info(log_info)
    print(log_info)


