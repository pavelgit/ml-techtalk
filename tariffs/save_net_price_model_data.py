from sources.DataProvider import DataProvider

data_provider = DataProvider()
examples = data_provider.load_from_db()

examples = list(filter(lambda example: example['input']['occupation'] == 'Erzieher,in' and example['output']['exists'], examples))

inputs = data_provider.read_examples_inputs_arrays_without_occupations(examples)
outputs = data_provider.read_examples_net_outputs_arrays(examples)

data_provider.save_to_file('net_price_model_inputs.npy', inputs)
data_provider.save_to_file('net_price_model_outputs.npy', outputs)