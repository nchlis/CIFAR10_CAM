# -*- coding: utf-8 -*-
"""
Created on Thu May  9 15:44:17 2019

@author: N.Chlis
"""

import numpy as np
import matplotlib.pyplot as plt
#from scipy.misc import toimage
#from keras.utils import np_utils
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from keras.models import Model
import cv2

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    #https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

from mpl_toolkits import axes_grid1
def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot.
    sources:
        - https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
        - https://nbviewer.jupyter.org/github/mgeier/python-audio/blob/master/plotting/matplotlib-colorbar.ipynb
    """
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

#%% load the data
#from keras.datasets import mnist
#(X_train, y_train), (X_test, y_test) = mnist.load_data()

from keras.datasets import cifar10
(X_tr, y_tr), (X_val, y_val) = cifar10.load_data()

#Constants for image dimensions
HEIGHT=32
WIDTH=32

#map numeric labels to class names
#source: https://www.cs.toronto.edu/~kriz/cifar.html
classes=np.unique(y_tr)
num_classes=len(classes)#number of classes
classes_char=np.array(['airplane','automobile','bird','cat','deer',
              'dog','frog','horse','ship','truck'])

#plot an example image from each class
fig=plt.figure(figsize=(3,6))
plt.suptitle('CIFAR 10 classes')
for i in np.arange(num_classes):
#    plt.figure(figsize=(30,30))
    pic=(np.where((y_tr==i))[0])[0]
    ax=fig.add_subplot(num_classes/2,2,i+1)
    ax.imshow(X_tr[pic,:,:])
    ax.set_title(str(i)+': '+classes_char[i])
fig.tight_layout()
fig.subplots_adjust(top=0.9)#to show suptitle properly
fig.show()
plt.savefig('./figures/CIFAR10_all_categories.png',dpi=300)

#normalize input images to [0,1]
X_tr=X_tr/2**8
X_val=X_val/2**8

#%% load the model and predict on validation set

model = load_model('CNN_CAM_128_256.hdf5')

#get predicted labels for the validation set
#y_val_hat = model.predict(X_val)
y_val_hat_prob = np.load('CNN_CAM_128_256_y_val_hat.npy')
print('shape of y_val_hat_prob:',y_val_hat_prob.shape)#(10000, 10)

#invert the to_categorical
y_val_hat=np.argmax(y_val_hat_prob,axis=1)
print('shape of y_val_hat:',y_val_hat.shape)#(10000,)

#elements of y_tr and y_val are encapsulated, need to clean up
print('y_val before:',y_val[0:10])
y_val = np.array([x[0] for x in y_val])
print('y_val after:',y_val[0:10])

#%% print confusion matrix

plot_confusion_matrix(y_val.tolist(), y_val_hat.tolist(), classes=classes_char,
                      title='Confusion matrix')

plt.savefig('./figures/CIFAR10_confusion_matrix.png',dpi=300)

#%% get CAM for an image/class pair.

def get_CAM(X,img):
    """
    get Class Activation Map (CAM) for a single image i in array X
    X: array of images with shape (nimages,height,width,channels)
    i: image to select from X
    returns CAM as an array of shape (height,width), normalized in [0,1].
    """
    X = np.expand_dims(X[img,:],axis=0)
    model_conv = Model(inputs=model.input, outputs=model.get_layer(name='activation_4').output)
    model_gap = Model(inputs=model.input, outputs=model.get_layer(name='GAP').output)
    
    C=model_conv.predict(X)[0]
    G=model_gap.predict(X)[0]
    
    CG=np.multiply(C,G)
    #sanity check to make sure np.multiply was used properly
    #for c in range(len(G)):
    #    np.array_equal(C[:,:,c]*G[c],CG[:,:,c])
    CAM=CG.sum(axis=-1)
    CAM=CAM/CAM.max()#normalize to 0-1, this step is not in the original CAM formula
    CAM=cv2.resize(CAM, dsize=(HEIGHT, WIDTH))#resize to original dimensions
    return(CAM)

#%% plot CAM for a single image

#cmap = plt.cm.coolwarm
cmap = plt.cm.seismic

i=4
CAM = get_CAM(X_val,i)

fig, axes = plt.subplots(1,2)
ax=axes[0]
ax.imshow(X_val[i,:])
ax=axes[1]
im=ax.imshow(CAM,cmap=cmap)
add_colorbar(im)
ax.imshow(X_val[i,:],alpha=0.4)

#%%
for code_class in range(len(classes_char)):
    ix_correct = np.where(y_val==y_val_hat)[0]#images the network gets right
    ix_wrong = np.where(y_val!=y_val_hat)[0]#images the network gets wrong
    print('Network accuracy:',len(ix_correct)/len(y_val))
    
    assert len(ix_correct)+len(ix_wrong)==len(y_val)#sanity check
    
    ix_class = np.where(y_val==code_class)[0]#images of selected class
    ix_correct_class = np.intersect1d(ix_correct,ix_class)#selected class images the network gets right
    ix_wrong_class = np.intersect1d(ix_wrong,ix_class)#selected class images the network gets wrong
    
    #% plot correct examles
    ix_plot = ix_correct_class[:5]
    nplots = len(ix_plot)
    fig, axes = plt.subplots(2,nplots,figsize=(3*nplots,3))
    for i in range(nplots):
        ax = axes[0,i]
        ax.imshow(X_val[ix_plot[i],:,:,:])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('class: '+classes_char[y_val_hat[ix_plot[i]]])
        ax = axes[1,i]
        CAM = get_CAM(X_val,ix_plot[i])
        im=ax.imshow(CAM,cmap=cmap)
        add_colorbar(im)
        ax.imshow(X_val[ix_plot[i],:,:,:],alpha=0.4)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig('./figures/CAM_'+classes_char[code_class]+'_correct.png', dpi=100, bbox_inches='tight')
    #plt.close()
    
    #% plot wrong examles
    ix_plot = ix_wrong_class[:5]    
    nplots = len(ix_plot)
    fig, axes = plt.subplots(2,nplots,figsize=(3*nplots,3))
    for i in range(nplots):
        ax = axes[0,i]
        ax.imshow(X_val[ix_plot[i],:,:,:])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('class: '+classes_char[y_val_hat[ix_plot[i]]])
        ax = axes[1,i]
        CAM = get_CAM(X_val,ix_plot[i])
        im=ax.imshow(CAM,cmap=cmap)
        add_colorbar(im)
        ax.imshow(X_val[ix_plot[i],:,:,:],alpha=0.4)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig('./figures/CAM_'+classes_char[code_class]+'_wrong.png',dpi=100, bbox_inches='tight')
    #plt.close()




