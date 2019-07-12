# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:48:39 2019

@author: N.Chlis
"""
#for the server ######
import matplotlib
matplotlib.use('Agg')
######################

import numpy as np
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split

#from scipy.misc import toimage
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import BatchNormalization
#from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.utils import np_utils
#from keras.models import load_model

#from keras.datasets import mnist
#(X_train, y_train), (X_test, y_test) = mnist.load_data()

from keras.datasets import cifar10
(X_tr, y_tr), (X_val, y_val) = cifar10.load_data()


#map numeric labels to class names
#source: https://www.cs.toronto.edu/~kriz/cifar.html
classes=np.unique(y_tr)
num_classes=len(classes)#number of classes
classes_char=['airplane','automobile','bird','cat','deer',
              'dog','frog','horse','ship','truck']

#plot an example image from each class
#fig=plt.figure(figsize=(3,6))
#plt.suptitle('CIFAR 10 classes')
#for i in np.arange(num_classes):
##    plt.figure(figsize=(30,30))
#    pic=(np.where((y_tr==i))[0])[0]
#    ax=fig.add_subplot(num_classes/2,2,i+1)
#    ax.imshow(X_tr[pic,:,:])
#    ax.set_title(str(i)+': '+classes_char[i])
#fig.tight_layout()
#fig.subplots_adjust(top=0.9)#to show suptitle properly
#fig.show()
#plt.savefig('CIFAR10_all_categories.png',dpi=300)

#normalize input images to [0,1]
X_tr=X_tr/2**8
X_val=X_val/2**8

#convert y to categorical
y_tr = np_utils.to_categorical(y_tr, num_classes)
y_val = np_utils.to_categorical(y_val, num_classes)

#%% set up the model

nfilters = [128, 256]
#nfilters = [64, 128]

model_id='CNN_CAM_'+str(nfilters[0])+'_'+str(nfilters[1])
print('Build model...',model_id)

model = Sequential()

#Conv block #1
model.add(Conv2D(nfilters[0], (3, 3), padding='same',
                 input_shape=X_tr.shape[1:]))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))

model.add(Conv2D(nfilters[0], (3, 3), padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

#Conv block #2
model.add(Conv2D(nfilters[1], (3, 3), padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))

model.add(Conv2D(nfilters[1], (3, 3), padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))#activation_4

#this is the CAM part: global average pooling and a dense layer with no bias
model.add(GlobalAveragePooling2D(name='GAP')) #output shape: (None, 256)
model.add(Dense(num_classes, use_bias=False, name = 'Dense'))#Zhou et al., 2016 use no bias.
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

#%% train the model

#log training progress
csvlog = CSVLogger(model_id+'_train_log.csv',append=True)

#save best model according to validation accuracy
checkpoint = ModelCheckpoint(model_id+'.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

#stop if no improvement on validation set
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)

max_epochs=100

hist=model.fit(X_tr, y_tr,
                 validation_data=(X_val, y_val),
                 epochs=max_epochs, batch_size=64, verbose=2,
                 initial_epoch=0,callbacks=[checkpoint, csvlog, early_stopping])
