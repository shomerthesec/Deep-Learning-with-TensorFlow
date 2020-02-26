#%% 1- train the model

from tensorflow.keras.models import Sequential , load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import optimizers
# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.
from tensorflow.keras.callbacks import TensorBoard , ModelCheckpoint
import pickle
import time

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0

dense_layers = [0]
layer_sizes = [64]
conv_layers = [3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME)

            model = Sequential()

            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Dropout(0.25))

            model.add(Flatten())

            for _ in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))
                model.add(Dropout(0.3))
                
            model.add(Dense(1))
            model.add(Activation('sigmoid'))
            
            checkpoint=ModelCheckpoint(filepath='checkpoint{}'.format(int(time.time())), monitor='val_loss',mode='auto', period=1)
            
            tensorboard = TensorBoard(log_dir='logs',batch_size=32, write_graph=True, write_grads=True)
            
            sgd=optimizers.SGD(lr=0.001, decay=1e-6 , momentum=0.9, nesterov=True)
            
            model.compile(loss='binary_crossentropy',
                          optimizer=sgd,
                          metrics=['accuracy'])

            model.fit(X, y,
                      batch_size=32,
                      epochs=15,
                      verbose=1,
                      validation_split= 0.3,
                      callbacks=[tensorboard, checkpoint])

model.save('64x3-CNN.model')

#%% 2- load model and do inference
import cv2 

# make sure to set the images dimentions as the dataset in the model also the order of the categories

CATEGORIES = ["Dog", "Cat"]  # will use this to convert prediction num to string value

def prepare(filepath):
    IMG_SIZE = 50  # 50 in modelxt
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # return the image with shaping that TF wants.
  

model =load_model("64x3-CNN.model")

#%% do your inference
prediction = model.predict([prepare('sample_test\dog.jpg')])  # REMEMBER YOU'RE PASSING A LIST OF THINGS YOU WISH TO PREDICT
print(prediction)
print(CATEGORIES[int(prediction[0][0])])

#also for the cat
prediction = model.predict([prepare('sample_test\cat.jpg')])
print(prediction)  # will be a list in a list.
print(CATEGORIES[int(prediction[0][0])],"suppose to be a cat")
