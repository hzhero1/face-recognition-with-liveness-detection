from liveness_net.livenessnet import LivenessNet
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
print(labels)
labels=np_utils.to_categorical(labels)
print(labels)
print(le.classes_)
print(len(le.classes_))
print(le)

image = cv2.imread("dataset/liveness/Fake/1012.png")
print(image)
print(image.shape)
image = cv2.resize(image, (32, 32))
print(image)

# imagePaths=list(paths.list_images("dataset/ROSE"))
# print(imagePaths)
# image=cv2.imread(imagePaths[1])
# image=np.array(image)
# print(image.shape)

