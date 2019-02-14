from sources.DataProvider import DataProvider
from sources.NoOccupationNetModelProvider import NoOccupationNetModelProvider
from keras.callbacks import TensorBoard
import time

data_provider = DataProvider()
inputs, outputs = data_provider.load_from_file()
train_inputs, dev_inputs = data_provider.split_examples(inputs)
train_outputs, dev_outputs = data_provider.split_examples(outputs)

model_provider = NoOccupationNetModelProvider()

tensor_board = TensorBoard(
    log_dir="logs/{}".format(time.strftime("%Y-%m-%dT%H-%M-%S")),
    histogram_freq=10
)

model_provider.model.fit(
    train_inputs,
    train_outputs,
    epochs=10000, batch_size=512,
    validation_data=(dev_inputs, dev_outputs),
    callbacks=[tensor_board]
)