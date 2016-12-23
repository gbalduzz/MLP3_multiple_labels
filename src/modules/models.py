from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D,Convolution3D
from keras.layers.convolutional import MaxPooling2D,MaxPooling3D,AveragePooling3D
from keras.utils import np_utils

num_classes = 3 # number of outputs

def main_model():
    # create model
    layer_depth = 30
    field = 5

    model = Sequential()
    model.add(Convolution3D(layer_depth, field, field, field, border_mode='valid', input_shape=(176, 208, 176),
        activation='relu',
        init='he_normal', W_constraint=maxnorm(3)))
    model.add(AveragePooling3D(pool_size=(4, 4, 4)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Convolution3D(30, 3, 3, 3, activation='relu', init='he_normal', W_constraint=maxnorm(2)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Convolution3D(40, 3, 3, 3, activation='relu', init='he_normal', W_constraint=maxnorm(2)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Convolution3D(50, 3, 3, 3, activation='relu', init='he_normal', W_constraint=maxnorm(2)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Convolution3D(50, 3, 3, 3, activation='relu', init='he_normal', W_constraint=maxnorm(2)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    #    model.add(Convolution3D(150, 1, 1, 1, activation='relu',init='he_normal', W_constraint = maxnorm(2)))
    #   model.add(Dropout(0.5))
    #    model.add(BatchNormalization())
    model.add(Convolution3D(100, 1, 1, 1, activation='relu', init='he_normal', W_constraint=maxnorm(2)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(num_classes, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def baseline_model():
    model = Sequential()
    model.add(Convolution3D(32, 3, 3, 3, border_mode='valid', input_shape=(176, 208, 176, 1),
                            activation='relu',init='he_normal',W_constraint = maxnorm(3)))
#    model.add(BatchNormalization())
    model.add(Flatten())
 #   model.add(Dense(128, activation='relu'))
 #   model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='relu'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
