from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras import backend as K
import keras
from keras.utils import multi_gpu_model
from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation, Add
from keras.layers import GlobalMaxPooling2D, GaussianDropout, BatchNormalization, ReLU
from keras.layers import GlobalAveragePooling2D, LeakyReLU, PReLU, SpatialDropout1D, SpatialDropout2D
from keras.layers.merge import add
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, ReduceLROnPlateau
from model import build_model

## Constants
width, height = 150, 150
classes = ['cizalla','mixto','rebaba','solido','vacio']

#Load Dataset

trainPath = 'Dataset/train'
validationPath = 'Dataset/validation'
testPath = 'Dataset/test'

#Instantiate ImageDataGenerator for train dataset
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


#Instantiate ImageDataGenerator for test dataset
test_datagen = ImageDataGenerator(rescale=1./255)

#Takes the path to a directory & generates batches of augmented data.
train_generator = train_datagen.flow_from_directory(
        trainPath,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validationPath,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical')


# Build the model

if K.image_data_format() == 'channels_first':
    input_shape = (3, width, height) 
else:
    input_shape = (width, height, 3)

ScrapModel = build_model(input_shape = input_shape, n_classes = 5)
ScrapModel.compile(optimizer = 'adadelta', loss = 'binary_crossentropy', metrics = ['acc'])
ScrapModel.summary()
# ScrapModel.fit(
#                x=x_train, y=y_train,epochs = 20, 
#                validation_data = (x_test, y_test), steps_per_epoch = 75,
#                validation_steps = 75
#                 )
ScrapModel.fit_generator(
        train_generator,
        steps_per_epoch=50,
        epochs=20,
        validation_steps = 75,
        validation_data=validation_generator)
#Fit the model
