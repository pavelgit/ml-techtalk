from sources.DataProvider import DataProvider
from sources.NoOccupationPresenceModelProvider import NoOccupationPresenceModelProvider

data_provider = DataProvider()
inputs, outputs = data_provider.load_from_file()

model_provider = NoOccupationPresenceModelProvider()

model_provider.model.fit(
    inputs,
    outputs,
    epochs=1000,
)