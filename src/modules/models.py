from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D,Convolution3D
from keras.layers.convolutional import MaxPooling2D,MaxPooling3D,AveragePooling3D
from keras.utils import np_utils
from keras.regularizers import l2, activity_l2

num_classes = 3 # number of outputs

def main_model(dims):
    # create model
    layer_depth = 64
    field = 5
    drop =0.2 # drop rate
    l2_w = 0.01 # l2 penalization rate

    model = Sequential()
    # first layer has wider field
    model.add(Convolution3D(layer_depth, field, field, field, border_mode='valid', input_shape=[dims[1], dims[2], dims[3], dims[4]],
        activation='relu', subsample=(2,2,2),
        init='he_normal', W_regularizer=l2(l2_w)))
#model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
    #model.add(Dropout(drop))
    #model.add(BatchNormalization())
    for i in range(3):
        layer_depth /= 2
        model.add(Convolution3D(layer_depth, 3, 3, 3, activation='relu', init='he_normal', W_regularizer=l2(l2_w)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
        #model.add(Dropout(drop))
        #model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(num_classes, activation='relu', W_regularizer=l2(l2_w)))

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
