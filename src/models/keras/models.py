from keras.models import Model

from keras.layers import Input
from keras.layers import Dense

from keras.layers import Conv2D
from keras.layers import Flatten


def baseline(fc_size=32, num_classes=10, conv_layers=4):
    """Baseline Implementation 
    """
    input_data = Input(name='inputs', shape=(256, 256, 3))
    x = input_data
    for n_layer in range(conv_layers):
        x = Conv2D(filters=fc_size, name='fc_{}'.format(n_layer+1),
                   kernel_size=3, padding='valid', 
                   activation='relu', strides=2)(x)
    x = Flatten()(x)
    output = Dense(num_classes, name="y_pred", activation='softmax')(x)
    model = Model(inputs=input_data, outputs=output)
    return model