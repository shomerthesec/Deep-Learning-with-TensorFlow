#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm


# In[2]:


DATADIR = 'CatsvsDogs'

CATEGORIES = ["Dog", "Cat"]


# In[12]:


# to modify the data set and prepare it 
training_data=[]
def create_training_data():
    for category in CATEGORIES:  # do dogs and cats

        path = os.path.join(DATADIR,category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat

        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
               # print("OSErrroBad img most likely", e, os.path.join(path,img))
           # except Exception as e:
               # print("general exception", e, os.path.join(path,img))

IMG_SIZE = 50
create_training_data()

print(type(training_data))
print(len(training_data))


# In[10]:


import random

random.shuffle(training_data)	#to shuffle the data


# creating the data set x for train y for test

X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


#Let's save this data, so that we don't need to keep calculating it every time we want to play with the neural network model:

import pickle

pickle_out = open("X.pickle","wb") # you can change the X.pickle as you want this is the name of the training set , wb for write
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()


# In[27]:


#We can always load it in to our current script, or a totally new one by doing:

pickle_in = open("X.pickle","rb") # rb for read
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)


# In[ ]:





