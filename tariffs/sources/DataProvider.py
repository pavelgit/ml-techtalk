from pymongo import MongoClient
import numpy as np
import pickle
import math 
import datetime

class DataProvider:

    NO_OCCUPATION_INPUT_ROW_LENGTH = 41

    FAMILY_STATUS_OPTIONS = [
        'Keine Angabe',
        'Paar mit Kind(ern)',
        'Single mit Kind(ern)',
        'Keine Kinder',
    ]

    EDUCATION_TYPE_OPTIONS = [
        'Promotion',
        'Studium (Master oder Diplom)',
        'Studium (Bachelor)',
        'Meister',
        'Berufsausbildung',
        'Abitur',
        'Realschulabschluss',
        'Hauptschulabschluss',
        'Kein Abschluss',
    ]

    JOB_SITUATION_OPTIONS = [
        'Angestellt/Selbstständig',
        'Beamter',
        'Sonstiges',
    ]

    INDUSTRY_OPTIONS = [
        'Elektro',
        'Gesundheitswesen',
        'Holz',
        'IT',
        'Kunststoff',
        'Metall',
        'Stahl',
        'Textil',
        'Sonstige Branche',
        'Sonstige',
    ]

    def load_from_db(self):
        with MongoClient('localhost', 27017) as client:
            db = client['bu-config']
            examples_collection = db['set_1151']
            count = examples_collection.find().count()
            print('Reading from database ', count)     
            progress = 0;       
            for example in examples_collection.find():
                yield example
                progress += 1
                if progress % 1000 == 0:
                    print('Progress', math.floor(1.0 * progress / count * 100.0))

    def save_to_file(self, file_name, array):
        np.save(file_name, array)

    def load_from_file(self, file_name):
        return np.load(file_name)

    def split_examples(self, inputs, outputs):
        assert len(inputs) == len(outputs)
        p = np.random.permutation(len(inputs))
        shuffled_inputs = inputs[p]
        shuffled_outputs = outputs[p]
        train_len = int(len(inputs) * 0.8)
        
        train_inputs = shuffled_inputs[0:train_len]
        train_outputs = shuffled_outputs[0:train_len]
        
        test_inputs = shuffled_inputs[train_len:]
        test_outputs = shuffled_outputs[train_len:]
        
        return train_inputs, train_outputs, test_inputs, test_outputs

    def read_examples_inputs_arrays_without_occupations(self, examples):
        return list(map(lambda example: list(self.get_example_input_array_without_occupation(example['input'])), examples))

    def read_examples_exists_outputs_arrays(self, examples):
        return list(map(lambda example: list(self._get_example_exists_output_array(example['output'])), examples))

    def read_examples_net_outputs_arrays(self, examples):
        return list(map(lambda example: list(self._get_example_net_output_array(example['output'])), examples))

    def get_example_input_array_without_occupation(self, example):
        yield from self._generate_one_hot(example['familyStatus'], self.FAMILY_STATUS_OPTIONS)
        yield from self._generate_one_hot(example['educationType'], self.EDUCATION_TYPE_OPTIONS)
        yield from self._generate_one_hot(example['jobSituation'], self.JOB_SITUATION_OPTIONS)
        yield from self._generate_one_hot(example['industry'], self.INDUSTRY_OPTIONS)

        yield example['benefitAgeLimit']
        yield example['benefitAmount']
        yield example['fractionOfficeWork']
        yield example['staffResponsibility']

        yield example['smoker']

        yield from self._generate_datetime_attributes(example['birthday'])
        yield from self._generate_datetime_attributes(example['insuranceStart'])

    def _generate_one_hot(self, value, possible_values):
        yield from (possible_value == value for possible_value in possible_values)

    def _generate_datetime_attributes(self, dt):
        yield dt.year
        yield dt.month
        yield dt.day
        yield dt.timetuple().tm_yday
        yield (dt - datetime.datetime(1970,1,1)).days

    def _get_example_output_array(self, example):
        return [
            example['exists'],
            example['net'],
            example['gross']
        ]

    def _get_example_exists_output_array(self, example):
        return [
            example['exists']
        ]

    def _get_example_net_output_array(self, example):
        return [
            example['net']
        ]
