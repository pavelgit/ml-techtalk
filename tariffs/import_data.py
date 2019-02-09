from sources.DataProvider import DataProvider

data_provider = DataProvider()
examples = data_provider.load_from_db()
inputs, outputs = data_provider.read_examples_arrays_without_occupations(examples)
data_provider.save_to_file(inputs, outputs)