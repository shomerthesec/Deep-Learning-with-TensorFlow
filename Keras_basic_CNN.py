#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D


# In[3]:


# importing the dataset

import pickle

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0 # to softly normalize the images with only dividing them by the max value 255 for colors


# In[4]:


model = Sequential()

model.add ( Conv2D(   128 , (3, 3) , input_shape=X.shape[1:]  ) )
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(  Conv2D(  64 , (3, 3) )  )
model.add(Activation('relu'))
model.add( MaxPooling2D( pool_size=(2, 2) ) )

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile( loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy']  )

model.fit(X, y, batch_size=32, epochs=3, validation_split=0.3)


# In[ ]:





