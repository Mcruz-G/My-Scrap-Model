
from keras import regularizers
from keras import backend as K
from keras.utils import multi_gpu_model
from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation, Add
from keras.layers import GlobalMaxPooling2D, GaussianDropout, BatchNormalization, ReLU
from keras.layers import GlobalAveragePooling2D, LeakyReLU, PReLU, SpatialDropout1D, SpatialDropout2D
from keras.layers.merge import add
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, ReduceLROnPlateau

def build_model(input_shape, n_classes):
    """
    input_shape : tuple
      Image Dims in a tuple
    n_classes : int
      Number of classes
    """
    
    model = Sequential()
    model.add(Conv2D(48, kernel_size=(3, 3), kernel_initializer='random_normal',
                     activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.6))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(128, (5, 5), activation='relu'))
    model.add(Dropout(0.7))
    
    model.add(MaxPooling2D(pool_size=(3, 3))) #added 25/12/2018
    model.add(Flatten())
    model.add(Dense(n_classes, activation='sigmoid'))

    return model

    