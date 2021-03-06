from keras.layers import Input, Dense, BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, load_model
from keras import optimizers
import keras.backend as K
from keras import regularizers
from sources.DataProvider import DataProvider
import keras.metrics

def max_abs_error(y_true, y_pred):
    return K.max(K.abs(y_true-y_pred))
    

class NoOccupationNetModelProvider:

    def __init__(self):
        self.model = None
        self.init_model()

    def init_model(self):
        x_input = Input((DataProvider.NO_OCCUPATION_INPUT_ROW_LENGTH,))

        x = x_input

        x = BatchNormalization()(x)
        x = Dense(
            1000, 
            activation='sigmoid',            
            kernel_regularizer=regularizers.l2(0.01), 
            activity_regularizer=regularizers.l1(0.01)
        )(x)

        x = Dense(1)(x)
        x = LeakyReLU()(x)

        self.model = Model(inputs=x_input, outputs=x)

        optimizer = optimizers.Adam(lr=0.01)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error', max_abs_error])

    def save(self):
        self.model.save('models/net_price_model.h5')

    def load(self):
        keras.metrics.max_abs_error = max_abs_error
        self.model = load_model('models/net_price_model.h5')
        self.model._make_predict_function()
