# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:04:43 2020

@author: ShomerTheSec
"""
#test

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.
from tensorflow.keras.callbacks import TensorBoard , ModelCheckpoint,EarlyStopping ,  ReduceLROnPlateau

from tensorflow.keras import optimizers
import time
from keras import backend as K



# In[]
img_width, img_height = 36, 36

train_data_dir = 'flowers/train'
validation_data_dir = 'flowers/valid'

nb_train_samples = 6552
nb_validation_samples = 1637
nb_classes = 102

batch_size = 64


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator( rescale=1. / 255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    vertical_flip= True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(  train_data_dir,
                                                      target_size=(img_width, img_height),
                                                      batch_size=batch_size,
                                                      class_mode='categorical',
                                                      shuffle=True)

validation_generator = test_datagen.flow_from_directory(  validation_data_dir,
                                                          target_size=(img_width, img_height),
                                                          batch_size=batch_size,
                                                          class_mode='categorical')


# In[2]:


NAME = "flowers-CNN-{}".format(int(time.time())) #very important to choose the name correctly to use tensorboard

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape = input_shape, padding='same'  ))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

print(model.summary())

# tensorboard = TensorBoard(log_dir='log',
#                           batch_size=128,
#                           write_graph=True,
#                           write_grads=True,
#                           write_images=True, 
#                           update_freq='batch') 
# tensorboard not working properly

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
              mode='min',
              restore_best_weights = True)

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                              factor = 0.2,
                              patience = 3,
                              verbose = 1,
                              min_delta = 0.0001)

sgd=optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])


# In[3]:

model.fit_generator(
                    train_generator,
                    steps_per_epoch=nb_train_samples ,
                    epochs=20, 
                    validation_data=validation_generator,
                    validation_steps=nb_validation_samples,
                    callbacks=[checkpoint,earlystop,reduce_lr])

# In[ ]:
model =load_model('checkpoints.epoch.08-val_loss.1.81.hdf5')


# In[4]

import json
with open('cat_to_name.json', 'r') as f:
           cat_to_name = json.load(f)
print (cat_to_name.values())

#%% confusion matrix and report
from sklearn.metrics import classification_report, confusion_matrix

#Confution Matrix and Classification Report
Y_pred = model.predict_generator(validation_generator, nb_validation_samples // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))
print('Classification Report')
target_names = list(cat_to_name.values())
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))

#%% plotting 
import matplotlib.pyplot as plt
import sklearn

img_row, img_height, img_depth = 36,36,3

class_labels = validation_generator.class_indices
class_labels = {v: k for k, v in class_labels.items()}
classes = list(cat_to_name.values())

#Confution Matrix and Classification Report
Y_pred = model.predict_generator(validation_generator, nb_validation_samples // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)

target_names = list(class_labels.values())

plt.figure(figsize=(20,20))
cnf_matrix = confusion_matrix(validation_generator.classes, y_pred)

plt.imshow(cnf_matrix, interpolation='nearest')
plt.colorbar()
tick_marks = np.arange( len(classes) )
_ = plt.xticks(tick_marks, classes, rotation=90)
_ = plt.yticks(tick_marks, classes)



# In[]
from vis.visualization import visualize_activation
from vis.utils import utils
from keras import activations
from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = (18, 6)

# Utility to search for layer index by name. 
# Alternatively we can specify this as -1 since it corresponds to the last layer.
layer_idx = utils.find_layer_idx(model, 'preds')

# Swap softmax with linear
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

# This is the output node we want to maximize.
filter_idx = 0
img = visualize_activation(model, layer_idx, filter_indices=filter_idx)
plt.imshow(img[..., 0])
