
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 16:09:21 2022

@author: palwi
"""


#!/usr/bin/env python
# coding: utf-8


#%%    
# Import libraries
# 
import tensorflow as tf
from tensorflow import keras
import os
import cv2  # pip install opencv-python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from skimage import io
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import (classification_report, accuracy_score, f1_score, recall_score, 
                          precision_score, confusion_matrix, roc_curve, auc)
import itertools
from statistics import mean, stdev
from itertools import cycle

# Keras-specific imports
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Dropout, Flatten, Activation, Input, Conv2D, BatchNormalization, 
                                MaxPool2D, GlobalAveragePooling2D)
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.utils import to_categorical  # Replace np_utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Sklearn imports
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

# from keras.backend import sigmoid
from keras.utils import get_custom_objects
#from keras.layers import Activation
##swish function
# def swish(x, beta = 1):
    # return (x * sigmoid(beta * x))

# get_custom_objects().update({'swish': Activation(swish)})
#leaky Rectified Linear Unit function
def leaky_relu(x):
    if x<0:
        return 0.01*x
    else:
        return x
    
get_custom_objects().update({'leaky relu': Activation(leaky_relu)})
#parameterized Rectified Linear Unit function
def parameterized_relu(x, a = 0.5):
    if x<0:
        return a*x
    else:
        return x
    
get_custom_objects().update({'parameterized relu': Activation(parameterized_relu)})
#Exponential Linear Unit function
def elu(x, a):
    if x<0:
        return a*(np.exp(x)-1)
    else:
        return x
    
get_custom_objects().update({'elu': Activation(elu)})

#%%
PATH = os.getcwd()
print(PATH)
# Define data path
data_path = PATH + '\\data path'
# data_path = PATH + '\\INbreast+MIAS+DDSM Dataset_3-class-with-AIIMS-data-aug-balanced'
# data_path = PATH + '\\INbreast+MIAS+DDSM_3-class'
# data_path = PATH + '\\INbreast+MIAS+DDSM_3-class_PNG_balanced'
# data_path = PATH + '\\INbreast+MIAS+DDSM Dataset-PNG'
# data_path = PATH + '\\INbreast+MIAS+DDSM+highly augumented AIIMS data'
# data_path = PATH + '\\highly augumented AIIMS data'
# data_path = PATH + '\\AIIMS augumented mammography dataset-new'
print(data_path)
data_dir_list = os.listdir(data_path)
print(data_dir_list)

img_rows=128
img_cols=128
num_channel=1
num_epoch=200

# Define the number of classes
num_classes = 3

img_data_list=[]

for dataset in data_dir_list:
    img_list=os.listdir(data_path+'\\'+ dataset)
    #print(img_list)
    print("......................")
    print(len(img_list))
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        input_img=cv2.imread(data_path + '\\'+ dataset + '\\'+ img)
        input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        # plt.imshow(input_img)
        input_img_resize=cv2.resize(input_img,(64,64))
        img_data_list.append(input_img_resize)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print("shape of image_data")
print (img_data.shape)

if num_channel==1:
    if K.image_data_format():#K.image_dim_ordering()=='th':
        img_data= np.expand_dims(img_data, axis=3)
        print (img_data.shape)
    else:
        img_data= np.expand_dims(img_data, axis=4) 
        print (img_data.shape)
        
else:
    if K.image_data_format():#K.image_dim_ordering()=='th':
        img_data=np.rollaxis(img_data,3,1)
        print (img_data.shape)
        
#%%
USE_SKLEARN_PREPROCESSING=False

if USE_SKLEARN_PREPROCESSING:
    # using sklearn for preprocessing
    from sklearn import preprocessing 
    
    def image_to_feature_vector(image, size=(128,128)):
        # resize the image to a fixed size, then flatten the image into
        # a list of raw pixel intensities
        return cv2.resize(image, size).flatten()
    
    img_data_list=[]
    for dataset in data_dir_list:
        img_list=os.listdir(data_path+'/'+ dataset)
        print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
        for img in img_list:
            input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
            input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            input_img_flatten=image_to_feature_vector(input_img,(128,128))
            img_data_list.append(input_img_flatten)
    
    img_data = np.array(img_data_list)
    img_data = img_data.astype('float32')
    print (img_data.shape)
    img_data_scaled = preprocessing.scale(img_data)
    print (img_data_scaled.shape)
    
    print (np.mean(img_data_scaled))
    print (np.std(img_data_scaled))
    
    print (img_data_scaled.mean(axis=0))
    print (img_data_scaled.std(axis=0))
    
    if K.image_data_format():#K.image_dim_ordering()=='th':
        img_data_scaled=img_data_scaled.reshape(img_data.shape[0],num_channel,img_rows,img_cols)
        print (img_data_scaled.shape)
        
    else:
        img_data_scaled=img_data_scaled.reshape(img_data.shape[0],img_rows,img_cols,num_channel)
        print (img_data_scaled.shape)
    
    
    if K.image_data_format():#K.image_dim_ordering()=='th':
        img_data_scaled=img_data_scaled.reshape(img_data.shape[0],num_channel,img_rows,img_cols)
        print (img_data_scaled.shape)
        
    else:
        img_data_scaled=img_data_scaled.reshape(img_data.shape[0],img_rows,img_cols,num_channel)
        print (img_data_scaled.shape)

if USE_SKLEARN_PREPROCESSING:
    img_data=img_data_scaled
#%%
# Assigning Labels

num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')
#print("Wakeel")

print("length of labels")
print(len(labels))

#%%
#for large COVID dataset
# labels[0:3674]=0
# labels[3675:5255]=1
# labels[5256:9505]=2

# labels[0:3615]=0
# labels[3616:5198]=1
# labels[5199:9471]=2

#for largest COVID dataset
# labels[0:7291]=0
# labels[7291:20340]=1
# labels[20340:25935]=2

#for COVID-19 largest dataset_all_PNG_balanced
# labels[0:5594]=0
# labels[5595:11189]=1
# labels[11190:16784]=2
#**********************************************

# for balanced_large COVID dataset
# labels[0:1580]=0
# labels[1581:3161]=1
# labels[3162:4742]=2

# labels[0:3615]=0
# labels[3616:5198]=1
# labels[5199:9471]=2

# for small_imbalanced dataset
# labels[0:299]=0
# labels[300:399]=1
# labels[400:799]=2
#%%#**********************************************
# for INbreast+MIAS+DDSM_3-class
labels[0:10866]=0
labels[10866:24576]=1
labels[24576:37410]=2

# for INbreast+MIAS+DDSM_3-class_PNG_balanced
# labels[0:10865]=0
# labels[10866:21731]=1
# labels[21732:32597]=2

# for INbreast+MIAS+DDSM Dataset-JPG_cut-down-balanced
# labels[0:5621]=0
# labels[5622:11243]=1
# labels[11244:16865]=2

# for INbreast+MIAS+DDSM Dataset_smallest
# labels[0:199]=0
# labels[200:419]=1
# labels[420:559]=2

# for INbreast+MIAS+DDSM Dataset_AIIMS_aug_balanced
# labels[0:11285]=0
# labels[11286:22571]=1
# labels[22572:33857]=2

# for AIIMS augumented mammography dataset
# labels[0:419]=0
# labels[420:629]=1
# labels[630:4824]=2

#for largest COVID dataset
# labels[0:7291]=0
# labels[7291:20340]=1
# labels[20340:25935]=2

# for AIIMS augumented mammography dataset
# labels[0:420]=0
# labels[420:630]=1
# labels[630:1244]=2

# for highly augumented AIIMS data
# labels[0:8372]=0
# labels[8373:16302]=1
# labels[16303:24697]=2

# for INbreast+MIAS+DDSM+highly augumented AIIMS data
# labels[0:19238]=0
# labels[19239:40879]=1
# labels[40880:59386]=2

print("labels:")
print(labels)
print("labels:")
print(labels.shape)
#names = ['covid','normal', 'pneumonia']
     
# convert class labels to on-hot encoding
#y = np_utils.to_categorical(labels, num_classes)

# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(labels)

#Shuffle the dataset
X,y = shuffle(img_data, labels)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=40)
#X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.79, test_size=0.21, random_state=2)
#print("trained labels")
#print(X_train)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
#                                                     random_state=0)

#%%
# Defining the model
'''
input_shape=img_data[0].shape
print("input shape")
print(input_shape)
model = Sequential()
model.build(input_shape)

#model.add(Conv2D(64, kernel_size=(3, 3), padding='same', input_shape=(32,128,128,1), data_format='channels_last'))
model.add(Conv2D(64, kernel_size=(3, 3), activation= swish, padding='same', input_shape=input_shape, data_format='channels_first'))
#model.add(Activation('relu'))
# model.add(Activation('swish'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
#model.add(Dropout(0.5))

model.add(Conv2D(128, kernel_size=(3, 3), activation= swish))
#model.add(Activation('relu'))
# model.add(Activation('swish'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
#model.add(Dropout(0.5))

model.add(Conv2D(256, kernel_size=(3, 3), activation= swish))
#model.add(Activation('relu'))
# model.add(Activation('swish'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.5))

model.add(Conv2D(512, kernel_size=(3, 3), activation= swish))
#model.add(Activation('relu'))
# model.add(Activation('swish'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.5))

# model.add(Conv2D(256, kernel_size=(3, 3)))
#model.add(Activation('relu'))
# model.add(Activation('swish'))
# model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
#model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation= swish))
#model.add(Activation('relu'))
# model.add(Activation('swish'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.5))

model.add(Flatten())
# model.add(Dense(128))
#model.add(BatchNormalization())
# model.add(Activation('swish'))
# model.add(Dropout(0.5))
# model.add(Dense(64))
# model.add(Activation('swish'))
model.add(Dense(64, activation= swish))
# model.add(Activation('swish'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation= 'softmax'))
# model.add(Activation('softmax'))
'''
'''
input_shape = img_data.shape[1:]
print(input_shape)
cnn_input = Input(shape=input_shape)
cnn_layers = Conv2D(16, kernel_size=(3, 3), activation='relu')(cnn_input)
cnn_layers = MaxPool2D(pool_size=(2, 2))(cnn_layers)
cnn_layers = Conv2D(32, kernel_size=(3, 3), activation='relu')(cnn_layers)
cnn_layers = MaxPool2D(pool_size=(2, 2))(cnn_layers)

cnn_layers = Conv2D(64, kernel_size=(3, 3), activation='relu')(cnn_layers)
cnn_layers = MaxPool2D(pool_size=(2, 2))(cnn_layers)
cnn_layers = Dropout(0.5)(cnn_layers)
cnn_layers = Conv2D(128, kernel_size=(3, 3), activation='relu')(cnn_layers)
cnn_layers = MaxPool2D(pool_size=(2, 2))(cnn_layers)
cnn_layers = Dropout(0.5)(cnn_layers)
# cnn_layers = Conv2D(256, kernel_size=(3, 3), activation='relu')(cnn_layers)
# cnn_layers = MaxPool2D(pool_size=(2, 2))(cnn_layers)
# cnn_layers = Dropout(0.5)(cnn_layers)
# cnn_layers = Conv2D(512, kernel_size=(3, 3), activation='swish')(cnn_layers)
# cnn_layers = MaxPool2D(pool_size=(2, 2))(cnn_layers)
# cnn_layers = Dropout(0.5)(cnn_layers)
# cnn_layers = Conv2D(128, kernel_size=(3, 3), activation='swish')(cnn_layers)
# cnn_layers = MaxPool2D(pool_size=(2, 2))(cnn_layers)
# cnn_layers = Dropout(0.5)(cnn_layers)

cnn_layers = Flatten()(cnn_layers)
cnn_layers = Dense(128, activation='relu')(cnn_layers)
# cnn_layers = Dense(128, activation='relu', kernel_regularizer=l1(0.001))(cnn_layers)
cnn_layers = BatchNormalization()(cnn_layers)
cnn_layers = Dropout(0.5)(cnn_layers)
cnn_layers = Dense(64, activation='relu', kernel_regularizer=l1(0.001))(cnn_layers)
cnn_layers = BatchNormalization()(cnn_layers)
cnn_layers = Dropout(0.5)(cnn_layers)
# cnn_layers = Dense(num_classes, activation='softmax')(cnn_layers)
cnn_layers = Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.001))(cnn_layers)
model = Model(inputs=cnn_input, outputs=cnn_layers)
'''
#%%

# Calculate class frequencies
class_counts = np.bincount(y_train)
print(class_counts)

# Automatically calculate class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
print(class_weights_dict)

def create_new_model(input_shape, num_classes):
    cnn_input = Input(shape=input_shape)
    cnn_layers = Conv2D(16, kernel_size=(3, 3), activation='relu')(cnn_input)
    cnn_layers = MaxPool2D(pool_size=(2, 2))(cnn_layers)
    cnn_layers = Conv2D(32, kernel_size=(3, 3), activation='relu')(cnn_layers)
    cnn_layers = MaxPool2D(pool_size=(2, 2))(cnn_layers)
    # cnn_layers = Dropout(0.5)(cnn_layers)
    cnn_layers = Conv2D(64, kernel_size=(3, 3), activation='relu')(cnn_layers)
    cnn_layers = MaxPool2D(pool_size=(2, 2))(cnn_layers)
    # cnn_layers = Dropout(0.5)(cnn_layers)
    # cnn_layers = DepthwiseConv2D(kernel_size=(3, 3), activation='relu')(cnn_layers)
    # cnn_layers = Conv2D(128, kernel_size=(3, 3), activation='relu')(cnn_layers)
    # cnn_layers = MaxPool2D(pool_size=(2, 2))(cnn_layers)
    cnn_layers = Dropout(0.5)(cnn_layers)
    # cnn_layers = GlobalAveragePooling2D()(cnn_layers)
    cnn_layers = Flatten()(cnn_layers)
    # cnn_layers = Dense(128, activation='relu')(cnn_layers)
    # cnn_layers = BatchNormalization()(cnn_layers)
    # cnn_layers = Dropout(0.5)(cnn_layers)
    # cnn_layers = Dense(128, activation='relu', kernel_regularizer=l1(0.001))(cnn_layers)
    cnn_layers = Dense(16, activation='relu', kernel_regularizer=l1(0.001))(cnn_layers)
    cnn_layers = BatchNormalization()(cnn_layers)
    cnn_layers = Dropout(0.5)(cnn_layers)
    cnn_layers = Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.001))(cnn_layers)
    
    model = Model(inputs=cnn_input, outputs=cnn_layers)
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Usage
input_shape = img_data.shape[1:]  # Assuming img_data is defined earlier
num_classes = 3  # Define the number of classes
# model = create_new_model(input_shape, num_classes)

#%%
'''

def create_modified_model(input_shape):
    model = models.Sequential()
    
    # First Convolutional Block
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Second Convolutional Block
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Third Convolutional Block
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Global Average Pooling
    model.add(layers.GlobalAveragePooling2D())
    
    # Fully Connected Layers
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))  # Adjust dropout rate
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(3, activation='softmax'))  # Output layer
    
    return model

# Compile the model with class weights
model = create_modified_model((128, 128, 1))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
'''
# model.summary() 
# Viewing model_configuration

'''
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape            
model.layers[0].output_shape            
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable
'''

#%%
#============================== Cross Validation =============================
# training start time
start = time.time()
# target_names = ['COVID-19', 'Normal', 'Pneumonia']
target_names = ['Benign', 'Malignant', 'Normal']    
accuracy = [ ]
precision = [ ]
f1Score = [ ]
sensitivity = [ ]
specificity = [ ]

colors = ['b', 'g', 'r', 'c', 'm']  # List of colors for each fold
plt.figure(figsize=(14, 6))
'''
# Plot Loss Curves
plt.subplot(1, 2, 1)
plt.title('Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Plot Accuracy Curves
plt.subplot(1, 2, 2)
plt.title('Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
'''
# Implement early stopping
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
skf = StratifiedKFold(n_splits=5, shuffle=True)
skf.get_n_splits(X_train, y_train)
foldNum=0
# for train_index, val_index in skf.split(X_train, y_train):
for (train_index, val_index), (fold, color) in zip(skf.split(X_train, y_train), zip(range(1, 6), colors)):
    foldNum+=1
    print("\nResults for fold",foldNum)
    model = create_new_model(input_shape, num_classes)

    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    # one hot encode
    y_train = keras.utils.to_categorical(y_train)
    y_val = keras.utils.to_categorical(y_val)
    # Initialize ImageDataGenerator
    datagen = ImageDataGenerator()

    # Create data generators for training and validation
    train_generator = datagen.flow(X_train, y_train, batch_size=32)
    val_generator = datagen.flow(X_val, y_val, batch_size=32)

    # Fit the model using generators
    hist = model.fit(train_generator, epochs=num_epoch, validation_data=val_generator)
    # hist = model.fit(X_train, y_train, class_weight=class_weights_dict, epochs=num_epoch, batch_size=128, validation_data=(X_val, y_val))
    # hist = model.fit(X_train, y_train, class_weight=class_weights_dict, validation_data = (X_val, y_val), epochs=num_epoch, batch_size=128)  
    # hist = model.fit(X_train, y_train, epochs=num_epoch, batch_size=128, validation_data=(X_val, y_val), callbacks=[early_stopping])
    model.summary()   
    # Create a new figure for each fold
    # plt.figure(figsize=(12, 5))
    '''
    # # Plot Loss Curves
    plt.subplot(1, 2, 1)
    plt.plot(hist.history['loss'], label=f'Fold {fold} Training Loss', color=color)
    plt.plot(hist.history['val_loss'], label=f'Fold {fold} Validation Loss', color=color, linestyle='dashed')

    # Plot Accuracy Curves
    plt.subplot(1, 2, 2)
    plt.plot(hist.history['accuracy'], label=f'Fold {fold} Training Accuracy', color=color)
    plt.plot(hist.history['val_accuracy'], label=f'Fold {fold} Validation Accuracy', color=color, linestyle='dashed')
    '''
    # Plot Loss Curves
    plt.figure(figsize=(7,5))
    plt.plot(hist.history['loss'], label='Train', color='blue')
    plt.plot(hist.history['val_loss'], label='Val', color='green')
    plt.title('train_loss vs val_loss')
    plt.xlabel('num of Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.style.use(['classic'])
    
    # Plot Accuracy Curves
    plt.figure(figsize=(7,5))
    plt.plot(hist.history['accuracy'], label='Train', color='blue')
    plt.plot(hist.history['val_accuracy'], label='Val', color='green')
    plt.title('train_acc vs val_acc')
    plt.xlabel('num of Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.style.use(['classic'])
    # Display the plot
    plt.show()
   
    yPredict = model.predict(X_val) 
    pred = np.argmax(yPredict, axis=1)
    val = np.argmax(y_val, axis=1)
    
    accuracy.append(accuracy_score(val, pred))
    precision.append(precision_score(val, pred, average='weighted'))
    f1Score.append(f1_score(val, pred, average='weighted')) 
    sensitivity.append(recall_score(val, pred, average='weighted'))
    print(classification_report(val, pred, target_names=target_names, digits=4))
    cm=confusion_matrix(val, pred)
    print(cm)
    # Calculate specificity
    specif = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity.append(specif)
'''
plt.subplot(1, 2, 1)
plt.legend(loc="upper right")
plt.subplot(1, 2, 2)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
'''
model.summary()    
print('\nList of possible accuracy:', accuracy)
print('Maximum Accuracy of model is:',
max(accuracy)*100, '%')
print('Minimum Accuracy of model is:',
min(accuracy)*100, '%')
print('OVERALL ACCURACY of model is:',
mean(accuracy)*100, '%')
# print('Standard Deviation for accuracy is:', stdev(accuracy))

print('\nList of possible precision:', precision)
print('Maximum precision of model is:',
max(precision)*100, '%')
print('Minimum precision of model is:',
min(precision)*100, '%')
print('OVERALL PRECISION of model is:',
mean(precision)*100, '%')
# print('Standard Deviation for precision is:', stdev(precision))

print('\nList of possible f1-score:', f1Score)
print('Maximum f1-score of model is:',
max(f1Score)*100, '%')
print('Minimum f1-score of model is:',
min(f1Score)*100, '%')
print('OVERALL F1-SCORE of model is:',
mean(f1Score)*100, '%')
# print('Standard Deviation for f1-score is:', stdev(f1Score))

print('\nList of possible sensitivity:', sensitivity)
print('Maximum sensitivity of model is:',
max(sensitivity)*100, '%')
print('Minimum sensitivity of model is:',
min(sensitivity)*100, '%')
print('OVERALL SENSITIVITY of model is:',
mean(sensitivity)*100, '%')
# print('Standard Deviation for sensitivity is:', stdev(sensitivity))

print('\nList of possible specificity:', specificity)
print('Maximum specificity of model is:',
max(specificity)*100, '%')
print('Minimum specificity of model is:',
min(specificity)*100, '%')
print('OVERALL SPECIFICITY of model is:',
mean(specificity)*100, '%')
# print('Standard Deviation for specificity is:', stdev(specificity))

end = time.time()
print('total training time', end - start)
#%%
#===============================Test Phase====================================
print("\nTest Results")
# testing start time
start = time.time()
yPred = model.predict(X_test)
y_test = keras.utils.to_categorical(y_test)
pred = np.argmax(yPred, axis=1)
test = np.argmax(y_test, axis=1)

accuracy = accuracy_score(test, pred)
precision=precision_score(test, pred, average='weighted')
f1Score=f1_score(test, pred, average='weighted') 
sensitivity = recall_score(test, pred, average='weighted')
# Compute confusion matrix
conf_matrix = (confusion_matrix(test, pred))
np.set_printoptions(precision=2)
specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
print(classification_report(test, pred, target_names=target_names, digits=4))
print("ACCURACY  : {}".format(accuracy))
print("PRECISION : {}".format(precision))
print("F1-SCORE : {}".format(f1Score))
print('sensitivity:', sensitivity)
print('specificity:', specificity)
# cm=confusion_matrix(test, pred)
# print(cm) 

end = time.time()
print('total testing time', end - start)
#%%
# ============================classwise accuracy==============================
# def classwise_accuracy():
a = pd.crosstab(test,pred)
print(a.max(axis=1)/a.sum(axis=1))
# classwise_accuracy()

#%%
'''
# ===============================Training=====================================
#hist = model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=1, validation_split=0.2, validation_data=(X_test, y_test))

hist = model.fit(X_train, y_train, batch_size=64, epochs=num_epoch, validation_split=0.2, verbose=2)
# Training with callbacks
# from keras import callbacks

# filename='model_train_new.csv'
# csv_log=callbacks.CSVLogger(filename, separator=',', append=False)

# early_stopping=callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='min')

# filepath="Best-weights-my_model-{epoch:03d}-{loss:.4f}-{accuracy:.4f}.hdf5"

# checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# callbacks_list = [csv_log,early_stopping,checkpoint]

#hist = model.fit(X_train, y_train, batch_size=64, epochs=num_epoch,verbose=1, validation_data=(X_test, y_test),callbacks=callbacks_list)
'''
# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['accuracy']
val_acc=hist.history['val_accuracy']

# xc=range(num_epoch)
xc=range(len(train_loss))
# xc = np.arange(len(train_loss))

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss, color = 'blue')
plt.plot(xc,val_loss, color = 'green')
# Set the x-axis ticks
# x_ticks = np.arange(0, num_epoch, 5)
# x_ticks = np.arange(0, xc, 5)  # Ticks from 0 to num_epoch with a step of 10
# plt.xticks(x_ticks)

plt.xlabel('num of Epochs')

# # Set the y-axis ticks
# y_ticks = np.arange(0, 0.21, 0.03)  # Ticks from 0 to 0.012 with a step of 0.002
# y_ticks = np.arange(0, 0.5, 0.05) 
# plt.yticks(y_ticks)

plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'], loc='upper right')
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc, color = 'blue')
plt.plot(xc,val_acc, color = 'green')
# Set the x-axis ticks
# x_ticks = np.arange(0, num_epoch, 5)
# x_ticks = np.arange(0, xc, 5)  # Ticks from 0 to num_epoch with a step of 10
# plt.xticks(x_ticks)

plt.xlabel('num of Epochs')

# # Set the y-axis ticks
# y_ticks = np.arange(0.94, 1.02, 0.02)  # Ticks from 0 to 0.012 with a step of 0.002
# y_ticks = np.arange(0.5, 1, 0.05) 
# plt.yticks(y_ticks)

plt.ylabel('accuracy')
plt.title('train_acc vs val_accuracy')
plt.grid(True)
plt.legend(['train','val'], loc='lower right')
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

'''
# Evaluating the model

# score = model.evaluate(X_test, y_test,verbose=2)
# print('Test Loss:', score[0])
# print('Test accuracy:', score[1])

# test_image = X_test[0:1]
# print(X_test[0:1])
# print (test_image.shape)

# print(model.predict(test_image))
# print(model.predict_classes(test_image))
# print(y_test[0:1])

# Testing
# print("hey")
# test_accu = model.evaluate("DBwhole_2_class - small_cov_norm//test/")
# print('The testing accuracy is :',test_accu[1]*100, '%')
# print("hi")

# Testing a new image
#test_image = io.imread('data//dataset//1//29.jpg')
test_image = cv2.imread('DBwhole_2_class - small_cov_norm//test/NORMAL/NORMAL2-IM-1828-0001 resized resized.jpg')
#test_image = cv2.imread('DBM//test1//person1676_virus_2892 resized resized.jpg')
test_image=cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
test_image=cv2.resize(test_image,(128,128))
test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255
#print (test_image.shape)
   
if num_channel==1:
    
    if K.image_data_format():#K.image_dim_ordering()=='th':
        test_image= np.expand_dims(test_image, axis=0)
        test_image= np.expand_dims(test_image, axis=0)
        #print (test_image.shape)
    else:
        test_image= np.expand_dims(test_image, axis=3) 
        test_image= np.expand_dims(test_image, axis=0)
        #print (test_image.shape)
        
else:
    if K.image_data_format():#K.image_dim_ordering()=='th':
        test_image=np.rollaxis(test_image,2,0)
        test_image= np.expand_dims(test_image, axis=0)
        #print (test_image.shape)
    else:
        test_image= np.expand_dims(test_image, axis=0)
        #print (test_image.shape)
        
# Predicting the test image
print("Test Image is\n")
print((model.predict(test_image)))
print("Test Image classes\n")
print(model.predict_classes(test_image))

#%%
# Visualizing the intermediate layer

def get_featuremaps(model, layer_idx, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()],[model.layers[layer_idx].output,])
    activations = get_activations([X_batch,0])
    return activations

layer_num=3
filter_num=0
activations = get_featuremaps(model, int(layer_num),test_image)

print (np.shape(activations))
feature_maps = activations[0][0]      
print (np.shape(feature_maps))

if K.image_data_format():#K.image_dim_ordering()=='th':
    feature_maps=np.rollaxis((np.rollaxis(feature_maps,2,0)),2,0)
print (feature_maps.shape)

fig=plt.figure(figsize=(16,16))
plt.imshow(feature_maps[:,:,filter_num],cmap='gray')
plt.savefig("featuremaps-layer-{}".format(layer_num) + "-filternum-{}".format(filter_num)+'.jpg')

num_of_featuremaps=feature_maps.shape[2]
fig=plt.figure(figsize=(16,16))    
plt.title("featuremaps-layer-{}".format(layer_num))
subplot_num=int(np.ceil(np.sqrt(num_of_featuremaps)))
for i in range(int(num_of_featuremaps)):
    ax = fig.add_subplot(subplot_num, subplot_num, i+1)
    #ax.imshow(output_image[0,:,:,i],interpolation='nearest' ) #to see the first filter
    ax.imshow(feature_maps[:,:,i],cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
plt.show()
fig.savefig("featuremaps-layer-{}".format(layer_num) + '.jpg')

'''
#%%
# testing a new image
'''
test_image = cv2.imread('resized test im2.jpg')
#test_image = cv2.imread('DBM//test1//person1676_virus_2892 resized resized.jpg')
test_image=cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
test_image=cv2.resize(test_image,(128,128))
test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255
#print (test_image.shape)
   
if num_channel==1:
    
    if K.image_data_format():#K.image_dim_ordering()=='th':
        test_image= np.expand_dims(test_image, axis=0)
        test_image= np.expand_dims(test_image, axis=0)
        #print (test_image.shape)
    else:
        test_image= np.expand_dims(test_image, axis=3) 
        test_image= np.expand_dims(test_image, axis=0)
        #print (test_image.shape)
        
else:
    if K.image_data_format():#K.image_dim_ordering()=='th':
        test_image=np.rollaxis(test_image,2,0)
        test_image= np.expand_dims(test_image, axis=0)
        #print (test_image.shape)
    else:
        test_image= np.expand_dims(test_image, axis=0)
        #print (test_image.shape)
        
# Predicting the test image
print("Test Image is\n")
print((model.predict(test_image)))
print("Test Image class\n")
print(model.predict_classes(test_image))
'''

# Printing the confusion matrix
'''
from sklearn.metrics import classification_report

Y_pred = model.predict(X_test)
#print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
#print(y_pred)
#y_pred = model.predict_classes(X_test)
#print(y_pred)
target_names = ['COVID-19', 'Normal']
                    
print(classification_report(np.argmax(y_test,axis=1), y_pred,target_names=target_names))

print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))

'''

# Plotting the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
# cnf_matrix = (confusion_matrix(test, pred))

np.set_printoptions(precision=2)

plt.figure()

# Plot non-normalized confusion matrix
plot_confusion_matrix(conf_matrix, classes=target_names,
                      title='Non-Normalized Confusion matrix')
plt.figure()
# Plot normalized confusion matrix
plot_confusion_matrix(conf_matrix, classes=target_names, normalize=True,
                      title='Normalized confusion matrix')
plt.figure()
plt.show()

print('y_test and yPred')
print(y_test.shape, yPred.shape)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], yPred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), yPred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curve
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average (AUC = {0:0.2f})'
               ''.format(roc_auc["micro"]), color="deeppink", linestyle=":", lw=1.5)

colors = cycle(['blue', 'red', 'green'])
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, label='Class {0} (AUC = {1:.2f})'
                                   ''.format(target_names[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], color='black', lw=1.5, linestyle='--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

#%%
'''
# Testing single image
test_image = cv2.imread('AIIMS data/AIIMS mammogram JPEG Images/PA001/21.jpg')
test_image=cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
test_image=cv2.resize(test_image,(128,128))
test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255
print(test_image.shape)
test_image_reshaped = np.reshape(test_image, (1, 128, 128, 1))
#print (test_image.shape)
# Predicting the test image
print("Test Image probability is ")
predicted_probability = model.predict(test_image_reshaped)
print(predicted_probability)
print("Test Image class is ")
# predicted_labels = np.argmax(predicted_probability, axis=1)
predicted_label = (predicted_probability >= 0.5).astype(int)
print(predicted_label)
'''

# Testing multiple images
print("-----------TESTING OF AIIMS DATASET----------")
folder_path = 'collective AIIMS 178 testing/'
# folder_path = 'test images/'
# List to hold predictions
predictions = []
test_img_list = []
# Iterate over the images in the folder
for filename in os.listdir(folder_path):
    # Read and preprocess the test image
    image_path = os.path.join(folder_path, filename)
    test_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    test_image = cv2.resize(test_image, (64, 64))
    test_img_list.append(test_image)

test_img = np.array(test_img_list)
test_img = test_img.astype('float32')
test_img /= 255
print("shape of test_img")
print (test_img.shape)

if num_channel==1:
    if K.image_data_format():#K.image_dim_ordering()=='th':
        test_img= np.expand_dims(test_img, axis=3)
        print (test_img.shape)
    else:
        test_img= np.expand_dims(test_img, axis=4) 
        print (test_img.shape)
        
else:
    if K.image_data_format():#K.image_dim_ordering()=='th':
        test_img=np.rollaxis(test_img,3,1)
        print (test_img.shape)

num_of_samples = test_img.shape[0]
T_labels = np.ones((num_of_samples,),dtype='int64')
#print("Wakeel")

print("length of labels")
print(len(T_labels))

# T_labels[0:13]=0    #13
# T_labels[14:17]=1   #5
# T_labels[18:59]=2   #40

T_labels[0:28]=0    #28
T_labels[28:38]=1   #10
T_labels[38:179]=2   #140

print("labels:")
print(T_labels)
print("labels:")
print(T_labels.shape)
#names = ['covid','normal', 'pneumonia']
     
# convert class labels to on-hot encoding
#y = np_utils.to_categorical(labels, num_classes)

# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(labels)

#Shuffle the dataset
X,y = shuffle(test_img, T_labels)

'''        
    prediction = model.predict(test_image_reshaped)
    predicted_class = np.argmax(prediction, axis=1)[0]  # Get the class index
        
    # Store the result
    predictions.append((filename, target_names[predicted_class]))

# Print predictions
for filename, class_name in predictions:
    print(f"{filename}: {class_name}")
    # Predict the test image
    # predicted_probability = model.predict(test_image_reshaped)
    # predicted_label = np.argmax(predicted_probability, axis=1)
    # predicted_label = (predicted_probability >= 0.5).astype(int)
    # Print the results
    # print(f"Image: {filename}")
    # print("Test Image probability is: ", predicted_probability)
    # print("Test Image class is: ", predicted_label)
    # print(f"Image: {filename} is predicted as:  ", predicted_label)
    # print("-----------------------------")
'''
yPrediction = model.predict(test_img)
y_testt = keras.utils.to_categorical(T_labels)
pred = np.argmax(yPrediction, axis=1)
test = np.argmax(y_testt, axis=1)

accuracy = accuracy_score(test, pred)
precision=precision_score(test, pred, average='weighted')
f1Score=f1_score(test, pred, average='weighted') 
sensitivity = recall_score(test, pred, average='weighted')
# Compute confusion matrix
conft_matrix = (confusion_matrix(test, pred))
np.set_printoptions(precision=2)
specificity = conft_matrix[0, 0] / (conft_matrix[0, 0] + conft_matrix[0, 1])
print(classification_report(test, pred, target_names=target_names, digits=4))
print("ACCURACY  : {}".format(accuracy))
print("PRECISION : {}".format(precision))
print("F1-SCORE : {}".format(f1Score))
print('sensitivity:', sensitivity)
print('specificity:', specificity)

# def classwise_accuracy():
a = pd.crosstab(test,pred)
print(a.max(axis=1)/a.sum(axis=1))

np.set_printoptions(precision=2)

plt.figure()

# Plot non-normalized confusion matrix
plot_confusion_matrix(conft_matrix, classes=target_names,
                      title='Non-Normalized Confusion matrix')
plt.figure()
# Plot normalized confusion matrix
plot_confusion_matrix(conft_matrix, classes=target_names, normalize=True,
                      title='Normalized confusion matrix')
plt.figure()
plt.show()

print('y_test and yPred')
print(y_testt.shape, yPrediction.shape)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_testt[:, i], yPrediction[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_testt.ravel(), yPrediction.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curve
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average (AUC = {0:0.2f})'
               ''.format(roc_auc["micro"]), color="deeppink", linestyle=":", lw=1.5)

colors = cycle(['blue', 'red', 'green'])
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, label='Class {0} (AUC = {1:.2f})'
                                   ''.format(target_names[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], color='black', lw=1.5, linestyle='--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
#%%
'''
# Saving and loading model and weights
from keras.models import model_from_json
from keras.models import load_model
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


model.save('model_try.h5')
# loaded_model=load_model('model.hdf5')
# Save only the model weights
model.save_weights('model_try_weights.h5')
'''