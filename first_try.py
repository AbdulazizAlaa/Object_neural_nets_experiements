#from __future__ import absolute_import
#from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import range
import cPickle as pickle
#from data import load_data

import os
from PIL import Image
import numpy as np



# Read folder mnist under 42,000 pictures, pictures to grayscale, so as a channel,
# If a color map as input, then a replacement is 3, and the data [i,:,:,:] = arr to Data [i,:,:,:] = [arr [:,:, 0], arr [:,:, 1], arr [:,:, 2]]
def  load_data():
    imgs = os.listdir("./train")
    num = len(imgs)
    data = np.empty((num,3,32,32),dtype="float32")
    label = np.empty((num,),dtype ="uint8")
    j=0
    for i in range(num):
        if not imgs[i].startswith('.'):
            label[j]=0 if imgs[i].split('.')[0]=='cat' else 1
            img = Image.open("./train/"+imgs[i])
            arr = np.asarray (img, dtype ="float32")
            if len(arr.shape) == 2:
                tmp = np.zeros((32, 32), dtype="float32")
                arr = [arr, tmp, tmp]
                arr = np.reshape(arr, (32,32,3))
            print j
            data [j,:,:,:] = [arr[:,:,0],arr[:,:, 1],arr[:,:, 2]]
            j=j+1
    return data, label



# Load data
# data,label = load_data()

##label 0 to 9 of 10 categories, keras requested format is binary class matrices, transforming it, directly call this function keras provided
# label = np_utils.to_categorical(label,2)

train_file = open('train.pkl', 'rb')
label_file = open('labels.pkl', 'rb')

# pickle.dump(data, train_file)
# pickle.dump(label, label_file)

data = pickle.load(train_file)
label = pickle.load(label_file)

train_file.close()
label_file.close()

print(data.shape[0],'samples')


batch_size = 32
nb_epoch = 1
#data_augmentation = True

# model = Sequential()

# model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, 32, 32)))
# model.add(Activation('relu'))
# model.add(Convolution2D(32, 3, 3))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
#
# model.add(Convolution2D(64, 3, 3, border_mode='same'))
# model.add(Activation('relu'))
# model.add(Convolution2D(64, 3, 3))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
#
# model.add(Flatten())
# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
#
# model.add(Dense(2))
# model.add(Activation('softmax'))

# # let's train the model using SGD + momentum (how original).
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy', optimizer=sgd)
# data=data.astype("float32")
# data/=255
# model.fit(data, label, batch_size=batch_size, nb_epoch=nb_epoch,shuffle=True,verbose=1,show_accuracy=True)
