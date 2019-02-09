from keras.layers import Input, Dense, BatchNormalization
from keras.layers.activations import Sigmoid
from keras.models import Model
from keras import optimizers
from sources.DataProvider import DataProvider


class SingleOccupationPresenceModelProvider:

    def __init__(self):
        self.model = None
        self.init_model()

    def init_model(self):
        x_input = Input((DataProvider.NO_OCCUPATION_INPUT_ROW_LENGTH,))

        x = x_input

        x = BatchNormalization()(x)
        x = Dense(5)(x)
        x = sigmoid()(x)

        x = Dense(3, activation='softmax')(x)

        self.model = Model(inputs=x_input, outputs=x)

        optimizer = optimizers.Adam(lr=0.01)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])


