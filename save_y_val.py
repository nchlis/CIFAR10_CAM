# -*- coding: utf-8 -*-
"""
Created on Thu May  9 15:53:01 2019

@author: N.Chlis
"""
from keras.models import load_model
import numpy as np

from keras.datasets import cifar10
(X_tr, y_tr), (X_val, y_val) = cifar10.load_data()

#normalize input images to [0,1]
X_tr=X_tr/2**8
X_val=X_val/2**8

model = load_model('CNN_CAM_128_256.hdf5')

#get predicted labels for the validation set
y_val_hat = model.predict(X_val)
np.save('CNN_CAM_128_256_y_val_hat.npy',y_val_hat)
