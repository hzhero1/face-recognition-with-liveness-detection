from pyimagesearch.livenessnet import LivenessNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import np_utils
from imutils import paths
#import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os

a=["fake", "fake", "real", "real", "fake"]
le=LabelEncoder()
labels=le.fit_transform(a)
labels=np_utils.to_categorical(labels)
print(labels)
print(le.classes_)
print(len(le.classes_))

imagePaths=list(paths.list_images("dataset1"))
print(imagePaths[1])
image=cv2.imread(imagePaths[1])
image=np.array(image)
print(image.shape)
