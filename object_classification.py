from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

import os
import cv2
import subprocess
from PIL import Image
import numpy as np

#
def getClass(classes, predictions):
    predictionsStr = ""
    for p, c in zip(predictions[0], classes):
        print p
        if p > .7:
            predictionsStr += c+", "
    return predictionsStr

def predictImages():
    # predictions
    size = (150, 150)

    filename = "./test1/"
    imgs = os.listdir(filename)
    num = len(imgs)
    for i in range(num):
        img = Image.open(filename+imgs[i])
        # arr = np.asarray (img, dtype ="float32")
        # print arr.shape

        img = img.resize(size, Image.ANTIALIAS)

        arr = np.asarray (img, dtype ="float32")
        # print arr.shape

        arr = np.reshape(arr, (1, 3, 150, 150))

        y_pred = model.predict(arr)

        print getClass(["aeroplane", "bicycle", "bird", "cat", "dog"], y_pred)

        cv2.imshow('img', cv2.imread(filename+imgs[i]))
        cv2.waitKey(0)

model = Sequential()

model.add(Convolution2D(32, 3, 3, input_shape=(3, 150, 150)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Activation('softmax'))

# model.load_weights("weights.h5")

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'data/train',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=32,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical')

model.fit_generator(
    train_generator,
    samples_per_epoch=10000,
    nb_epoch=100,
    validation_data=validation_generator,
    nb_val_samples=1000)

# predictImages()

model.save_weights('weights.h5')  # always save your weights after training or during training
