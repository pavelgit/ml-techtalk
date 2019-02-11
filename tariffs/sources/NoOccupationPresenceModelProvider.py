from keras.layers import Input, Dense, BatchNormalization, Activation
from keras.models import Model
from keras import optimizers
from sources.DataProvider import DataProvider


class NoOccupationPresenceModelProvider:

    def __init__(self):
        self.model = None
        self.init_model()

    def init_model(self):
        x_input = Input((DataProvider.NO_OCCUPATION_INPUT_ROW_LENGTH,))

        x = x_input

        x = BatchNormalization()(x)
        x = Dense(100)(x)
        x = Dense(100)(x)
        x = Dense(1, activation='sigmoid')(x)

        self.model = Model(inputs=x_input, outputs=x)

        optimizer = optimizers.Adam(lr=0.01)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


