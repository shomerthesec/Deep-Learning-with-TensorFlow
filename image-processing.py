from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K




img_width, img_height = 50, 50

train_data_dir = 'flowers/train'
validation_data_dir = 'flowers/valid'

nb_train_samples = 6552
nb_validation_samples = 1637


batch_size = 32

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
                                                      class_mode='binary')

validation_generator = test_datagen.flow_from_directory(  validation_data_dir,
                                                          target_size=(img_width, img_height),
                                                          batch_size=batch_size,
                                                          class_mode='binary')


                                                        
#%% how to fit the model
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)
