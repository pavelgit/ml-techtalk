from keras.layers import Input, Dense, BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras import optimizers
from sources.DataProvider import DataProvider

class NoOccupationNetModelProvider:

    def __init__(self):
        self.model = None
        self.init_model()

    def init_model(self):
        x_input = Input((DataProvider.NO_OCCUPATION_INPUT_ROW_LENGTH,))

        x = x_input

        x = BatchNormalization()(x)
        x = Dense(500)(x)
        x = LeakyReLU()(x)

        x = BatchNormalization()(x)
        x = Dense(500)(x)
        x = LeakyReLU()(x)

        x = BatchNormalization()(x)
        x = Dense(200)(x)
        x = LeakyReLU()(x)

        x = BatchNormalization()(x)
        x = Dense(100)(x)
        x = LeakyReLU()(x)

        x = Dense(1)(x)
        x = LeakyReLU()(x)

        self.model = Model(inputs=x_input, outputs=x)

        optimizer = optimizers.Adam(lr=0.01)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])


