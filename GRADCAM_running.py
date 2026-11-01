# -*- coding: utf-8 -*-
"""
Created on Wed May 14 12:42:51 2025

@author: palwi
"""

# -*- coding: utf-8 -*-
"""
Improved Grad-CAM Implementation
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
import os, cv2
from sklearn.utils import shuffle
from tensorflow.keras import Input, Model
from tensorflow.keras.utils import to_categorical

# Step 1: Define the Model
def create_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # Convolutional and pooling layers
    x = layers.Conv2D(16, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Flatten and dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation='relu')(x)
    outputs = layers.Dense(3, activation='softmax')(x)
    
    return Model(inputs=inputs, outputs=outputs)

# Step 2: Prepare the Data
PATH = os.getcwd()
print(PATH)
# Define data path
data_path = PATH + '\\data path'
# data_path = PATH + '\\AIIMS augumented mammography dataset-new'
print(data_path)
data_dir_list = os.listdir(data_path)
print(data_dir_list)

img_rows=128
img_cols=128
num_channel=1
num_epoch=30

# Define the number of classes
num_classes = 3

# data preprocessing
img_data_list=[]

for dataset in data_dir_list:
    img_list=os.listdir(data_path+'\\'+ dataset)
    #print(img_list)
    print("......................")
    print(len(img_list))
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        input_img=cv2.imread(data_path + '\\'+ dataset + '\\'+ img )
        input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img_resize=cv2.resize(input_img,(128,128))
        img_data_list.append(input_img_resize)

# if input_img is None:
#     print("Failed to load image")
# else:
#     print("Image shape:", input_img.shape)
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

# Assigning Labels

num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')
#print("Wakeel")

print("length of labels")
print(len(labels))
# for INbreast+MIAS+DDSM Dataset_smallest
# labels[0:420]=0
# labels[420:630]=1
# labels[630:1244]=2

# for AIIMS augumented mammography dataset
labels[0:420]=0
labels[420:630]=1
labels[630:1244]=2

# Convert labels to one-hot encoding
labels = to_categorical(labels, num_classes=3)

#Shuffle the dataset
X,y = shuffle(img_data, labels)

# Step 3: Compile and Train the Model 
input_shape = (128, 128, 1)  # Grayscale image
model = create_model(input_shape)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()  # Look for the name of your last Conv2D layer

# class_weights = {0: 1.0, 1: 1.0, 2: 5.0}  # Adjust weights based on class distribution
# model.fit(X, y, epochs=5, batch_size=16, validation_split=0.2, class_weight=class_weights)
# Train the model on the dummy dataset
model.fit(X, y, epochs=30, batch_size=32, validation_split=0.2)

# Step 4: Improved Grad-CAM Implementation
def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    # Create model that maps the input image to the activations
    # of the last conv layer and the model predictions
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # Compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(np.expand_dims(img_array, axis=0))
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    # Gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # Pool the gradients over all the axes leaving out the channel dimension
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # For visualization purpose, normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = cv2.resize(heatmap.numpy(), (128, 128))  # Resize to match input

    return heatmap

def display_gradcam(img, heatmap, alpha=0.4):
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)
    
    # Use jet colormap to colorize heatmap
    jet = plt.cm.get_cmap("jet")
    
    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    
    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    
    # Superimpose the heatmap on original image
    if len(img.shape) == 2 or img.shape[2] == 1:  # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 3:  # Already RGB
        pass
    else:  # Other cases
        img = img[:, :, :3]
    
    # Ensure both images are float32 in range [0, 1]
    img = img.astype(np.float32) / 255
    jet_heatmap = jet_heatmap.astype(np.float32) / 255
    
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = superimposed_img / np.max(superimposed_img)
    
    # Create a figure with 3 subplots: Original, Grad-CAM, and Colorbar
    plt.figure(figsize=(10, 5))
    
    # Subplot 1: Original Image
    plt.subplot(1, 3, 1)
    if len(img.shape) == 2 or img.shape[2] == 1:  # Grayscale
        plt.imshow(img.squeeze(), cmap='gray')
    else:  # RGB
        plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')
    
    # Subplot 2: Grad-CAM Heatmap
    plt.subplot(1, 3, 2)
    # heatmap_display = plt.imshow(heatmap, cmap='jet')
    heatmap_display = plt.imshow(superimposed_img)
    plt.title("Grad-CAM Heatmap")
    plt.axis('off')
    
    # Subplot 3: Colorbar
    # plt.subplot(1, 3, 3)
    plt.colorbar(heatmap_display, fraction=0.046, pad=0.04)
    # plt.title("Intensity Scale")
    # plt.axis('off')  # Hide axes for colorbar subplot
    
    plt.tight_layout()
    plt.show()
'''
def display_gradcam(img, heatmap, alpha=0.4):
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)
    
    # Use jet colormap to colorize heatmap
    jet = plt.cm.get_cmap("jet")
    
    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    
    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    
    # Superimpose the heatmap on original image
    if len(img.shape) == 2 or img.shape[2] == 1:  # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 3:  # Already RGB
        pass
    else:  # Other cases
        img = img[:, :, :3]
    
    # Ensure both images are float32 in range [0, 1]
    img = img.astype(np.float32) / 255
    jet_heatmap = jet_heatmap.astype(np.float32) / 255
    
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = superimposed_img / np.max(superimposed_img)
    
    # Display
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img.squeeze(), cmap='gray' if len(img.shape) == 2 else None)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img)
    plt.title("Grad-CAM")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
'''
# Step 5: Use Grad-CAM on a Sample Image
sample_image = X[552]  # Select an image
print("Sample image shape:", sample_image.shape)  # Should be (128, 128, 1)
# Get the name of the last Conv2D layer automatically
last_conv_layer_name = [layer.name for layer in model.layers if isinstance(layer, layers.Conv2D)][-1]
print("Last conv layer name:", last_conv_layer_name)

# Get model's prediction
# preds = model.predict(np.expand_dims(sample_image, axis=0))
# print("Model predictions:", preds)
predictions = model.predict(np.array([sample_image]))
print(predictions)

# Generate heatmap
heatmap = get_gradcam_heatmap(model, sample_image, last_conv_layer_name)
print("Heatmap min/max:", heatmap.min(), heatmap.max())  # Should be 0.0 to 1.0
# Display results
print(sample_image.shape)
print(heatmap.shape)
print(sample_image.dtype)
print(heatmap.dtype)
print(np.unique(heatmap))
# print(cv2.__version__)
# Display Grad-CAM
display_gradcam(sample_image, heatmap)
# display_gradcam(sample_image.squeeze(), heatmap)

'''
def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    # Create gradient model targeting the last conv layer and output
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(np.expand_dims(img_array, axis=0))
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Compute gradients
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Generate raw heatmap (shape: [conv_height, conv_width])
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)  # Shape: (height, width)

    # Normalize and upsample to (128, 128)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)  # Range [0, 1]
    heatmap = cv2.resize(heatmap.numpy(), (128, 128))  # Resize to match input

    return heatmap  # Shape: (128, 128)

def display_gradcam(img, heatmap, alpha=0.4):
    # Convert grayscale image to RGB if needed
    if len(img.shape) == 2:  # Shape (128, 128)
        img = np.stack((img,) * 3, axis=-1)  # Convert to (128, 128, 3)
    elif img.shape[2] == 1:  # Shape (128, 128, 1)
        img = cv2.cvtColor(img.squeeze(), cv2.COLOR_GRAY2RGB)

    # Normalize heatmap to [0, 255] and apply colormap
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose heatmap on original image
    superimposed_img = cv2.addWeighted(img, 1-alpha, heatmap_colored, alpha, 0)

    # Display
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img)
    plt.title("Grad-CAM")
    plt.axis('off')
    plt.show()

# Usage
sample_image = X[423]  # Shape: (128, 128, 1)
last_conv_layer_name = [layer.name for layer in model.layers if isinstance(layer, layers.Conv2D)][-1]
heatmap = get_gradcam_heatmap(model, sample_image, last_conv_layer_name)
print("Heatmap shape:", heatmap.shape)  # Should be (128, 128)

display_gradcam(sample_image.squeeze(), heatmap)
'''