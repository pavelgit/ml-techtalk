from sources.DataProvider import DataProvider
from sources.NoOccupationNetModelProvider import NoOccupationNetModelProvider
from keras.callbacks import TensorBoard
import time

data_provider = DataProvider()

inputs = data_provider.load_from_file('net_price_model_inputs.npy')
outputs = data_provider.load_from_file('net_price_model_outputs.npy')

train_inputs, train_outputs, test_inputs, test_outputs = data_provider.split_examples(inputs, outputs)

model_provider = NoOccupationNetModelProvider()

tensor_board = TensorBoard(
    log_dir="logs/{}".format(time.strftime("%Y-%m-%dT%H-%M-%S")),
    histogram_freq=10
)

model_provider.model.fit(
    train_inputs,
    train_outputs,
    epochs=100000, batch_size=512,
    validation_data=(test_inputs, test_outputs),
    callbacks=[tensor_board]
)
