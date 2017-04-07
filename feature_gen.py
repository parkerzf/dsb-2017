from keras.layers import Flatten, Input
from keras.layers import AveragePooling3D, MaxPooling3D

from keras.models import Model

from keras import backend as K

import numpy as np


def generate_spatial_agg_features(X, input_shape=(11, 11, 11, 256)):
    img_input = Input(shape=input_shape)

    x = MaxPooling3D((3, 3, 3), strides=(3, 3, 3), name='block1_pool')(img_input)
    # x = AveragePooling3D((3, 3, 3), strides=(3, 3, 3), name='block1_pool')(img_input)
    x = Flatten(name='flatten')(x)

    model = Model(inputs=img_input, outputs=x)

    res = model.predict(X)
    print res.shape


K.set_image_data_format('channels_last')

X = np.load('0121c2845f2b7df060945b072b2515d7.npy')
X = X.reshape(1, 11, 11, 11, 256)

generate_spatial_agg_features(X)







