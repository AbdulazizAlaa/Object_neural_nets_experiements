from shutil import copyfile
import pandas as pd
import numpy as np

src_filename_train = "/home/abdulaziz/Downloads/datasets/VOCdevkit/VOC2012/ImageSets/Main/bird_train.txt"
src_filename_trainval = "/home/abdulaziz/Downloads/datasets/VOCdevkit/VOC2012/ImageSets/Main/bird_trainval.txt"
src_filename_val = "/home/abdulaziz/Downloads/datasets/VOCdevkit/VOC2012/ImageSets/Main/bird_val.txt"

src_foldername = "/home/abdulaziz/Downloads/datasets/VOCdevkit/VOC2012/JPEGImages/"

dest_foldername_train = "/home/abdulaziz/Downloads/datasets/cats_dogs_classifier/data/train/bird/"
dest_foldername_val = "/home/abdulaziz/Downloads/datasets/cats_dogs_classifier/data/validation/bird/"

train = pd.read_csv(src_filename_train, sep=' ', error_bad_lines=False)
train_val = pd.read_csv(src_filename_trainval, sep=' ', error_bad_lines=False)
val = pd.read_csv(src_filename_val, sep=' ', error_bad_lines=False)

j=0
for i in range(train.shape[0]):
    if train.iloc[i,1]==1:
        filename = train.iloc[i,0]
        copyfile(src_foldername+filename+".jpg", dest_foldername_val+filename+".jpg")
        # print filename
        j+=1
print j

j=0
for i in range(train_val.shape[0]):
    if train_val.iloc[i,1]==1:
        filename = train_val.iloc[i,0]
        copyfile(src_foldername+filename+".jpg", dest_foldername_train+filename+".jpg")
        # print filename
        j+=1

print j

j=0
for i in range(val.shape[0]):
    if val.iloc[i,1]==1:
        filename = val.iloc[i,0]
        copyfile(src_foldername+filename+".jpg", dest_foldername_train+filename+".jpg")
        # print filename
        j+=1

print j
