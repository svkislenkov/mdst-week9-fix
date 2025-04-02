import tensorflow as tf
import os
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


##! ARE YOU A WINDOWS USER OR HAVE A GPU ON YOUR COMPUTER? UNCOMMENT THIS CELL!!!!!!!!!!
# (ew)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)

tf.config.list_physical_devices('GPU')

#TODO: Change 'data' to the name of your training set directory
# You should see a list of classes
data_dir = 'fruits-360_dataset_100x100/fruits-360/Training'
os.listdir(data_dir)

img = mpimg.imread('/home/evanteal15/mdst/mdst-classifer-starter/mdst_fruit_classifier/classifier/fruits-360_dataset_100x100/fruits-360/Training/Apple Braeburn 1/r_1_100.jpg')
print(img.shape)
plt.imshow(img)

# This formats our data...
# TODO: Ensure the image size is kept at (100, 100)
data = tf.keras.utils.image_dataset_from_directory(data_dir, image_size=(100, 100), shuffle=True)

# Each time we call this, it gives us a new set of data
data_iterator = data.as_numpy_iterator()

# 32 images per batch, 100x100, 3 channels (R, G, B)
batch = data_iterator.next()
batch[0].shape

# Hint: Pixel values range from 0-255. We want to scale x to range between 0-1.
# x represents our data, and y represents our class. Therefore, we shouldn't worry about y

# TODO: Uncomment + complete the following statement:
data = data.map(lambda x,y: (x/255, y))

# This will now give us an iterator with our SCALED data!
scaled_iterator = data.as_numpy_iterator()
batch = scaled_iterator.next()

# Once previous TODOs are complete, you should see 4 100x100 images here (of fruits, hopefully)
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(batch[1][idx])

test_data_dir = 'fruits-360_dataset_100x100/fruits-360/Test'
os.listdir(test_data_dir)

# This formats our data...
# TODO: Ensure the image size is kept at (100, 100)
test_data = tf.keras.utils.image_dataset_from_directory(test_data_dir)

# # Each time we call this, it gives us a new set of data
# data_iterator = data.as_numpy_iterator()

# # 32 images per batch, 100x100, 3 channels (R, G, B)
# batch = data_iterator.next()
# batch[0].shape

test_data = test_data.map(lambda x,y: (x/255, y))

# This will now give us an iterator with our SCALED data!
scaled_iterator = test_data.as_numpy_iterator()
batch = scaled_iterator.next()

# Once previous TODOs are complete, you should see 4 100x100 images here (of fruits, hopefully)
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(batch[1][idx])

train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.3)
# Leave our test data alone

# TODO: put in the name of your test_data here
test_size = int(len(test_data))

# TODO: Make sure train_size + val_size + test_size lines up with the total size of your data...
print(train_size + val_size + test_size)

# Notice how we separate the training + validation data...
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = test_data.take(test_size)