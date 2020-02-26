#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf  # deep learning library. Tensors are just multi-dimensional arrays

mnist = tf.keras.datasets.mnist  # mnist is a dataset of 28x28 images of handwritten digits and their labels
(x_train, y_train),(x_test, y_test) = mnist.load_data()  # unpacks images to x_train/x_test and labels to y_train/y_test

x_train = tf.keras.utils.normalize(x_train, axis=1)  # scales data between 0 and 1
x_test = tf.keras.utils.normalize(x_test, axis=1)  # scales data between 0 and 1


# In[3]:


model = tf.keras.models.Sequential()  # a basic feed-forward model
model.add(tf.keras.layers.Flatten())  # takes our 28x28 and makes it 1x784
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  # our output layer. 10 units for 10 classes. Softmax for probability distribution


# In[4]:


model.compile(optimizer='adam',  # Good default optimizer to start with
              loss='sparse_categorical_crossentropy',  # how will we calculate our "error." Neural network aims to minimize loss.
              metrics=['accuracy'])  # what to track


# In[5]:


model.fit(x_train, y_train, epochs=3)  # train the model


# In[10]:


val_loss, val_acc = model.evaluate(x_test, y_test)  # evaluate the out of sample data with model
print('the validation loss is :' , val_loss)  # model's loss (error)
print('the validation accuracy is :' , val_acc)  # model's accuracy


# In[7]:


model.save('epic_num_reader.model') # to save the model


# In[8]:


new_model = tf.keras.models.load_model('epic_num_reader.model') # to load the model


# In[17]:


predictions = new_model.predict(x_test) # to make predictions
import numpy as np
import matplotlib.pyplot as plt
print('the predicted number is :' , np.argmax(predictions[0])) # to take the highest probability 

plt.imshow(x_test[0],cmap=plt.cm.binary)  # to see the test pic
plt.show()


# In[ ]:





