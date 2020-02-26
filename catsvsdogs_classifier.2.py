#!/usr/bin/env python
# coding: utf-8

#%%Loading our libraries

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping #,TensorBoard  

import pickle
from tensorflow.keras import optimizers
import time

#%% Uploading our data
pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0

#%% building our model archeticture 

dense_layers = [256]
layer_sizes = [64,128]
conv_layers = [2]
NAME = "Cats-vs-dogs-{}-CNN-layers-sized{}-and-Denselayer-sized-{}-{}".format(conv_layers,layer_sizes , dense_layers ,int(time.time())) #very important to choose the name correctly to use tensorboard

model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(1))
model.add(Activation('sigmoid'))

# tensorboard = TensorBoard(log_dir='log',
#                           batch_size=128,
#                           write_graph=True,
#                           write_grads=True,
#                           write_images=True, 
#                           update_freq='batch') 
# tensorboard not working properly

#%% defining our extra functions to help us improve our training 
checkpoint=ModelCheckpoint(filepath='checkpoints.epoch.{epoch:02d}-val_loss.{val_loss:.2f}.hdf5',
                           monitor='val_loss',
                           verbose=1,
                           save_best_only=False,
                           save_weights_only=False,
                           mode='auto',
                           period=1)
earlystop=EarlyStopping(monitor='val_loss',
              verbose=1,
              patience=3,
              mode='min')

sgd=optimizers.SGD( lr=0.01, 
                   decay=1e-6,
                   momentum=0.9,
                   nesterov=True) 
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])


#%% fitting/training the model


model.fit(X, y,
          batch_size=32,
          epochs=20,
          validation_split=0.3,
          callbacks=[checkpoint,earlystop])


#%% in case of needing to load the model without training 
model =load_model('M:/Courses/Keras and Tensorflow/CNN.with.earlystop.and.checkpoint\checkpoints.epoch.06-val_loss.0.47.hdf5')


#%% doing the predictions or inference
import cv2
CATEGORIES=['DOG','CAT']
def prepare(filepath):
    IMG_SIZE = 50  # 50 in modelxt
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # return the image with shaping that TF wants.
prediction = model.predict([prepare('sample_test\dog.jpg')])  # REMEMBER YOU'RE PASSING A LIST OF THINGS YOU WISH TO PREDICT
print(prediction)
print(CATEGORIES[int(prediction[0][0])])

#also for the cat
prediction = model.predict([prepare('sample_test\cat.jpg')])
print(prediction)  # will be a list in a list.
print(CATEGORIES[int(prediction[0][0])],"suppose to be a cat")


