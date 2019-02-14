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

    def save_to_file(self, input_np_array, output_np_array):
        np.save('input_np_array.npy', input_np_array)
        np.save('output_np_array.npy', output_np_array)

    def load_from_file(self):
        input_np_array = np.load('input_np_array.npy')
        output_np_array = np.load('output_np_array.npy')
        return input_np_array, output_np_array

    def split_examples(self, examples):
        train_len = int(len(examples) * 0.8)
        return examples[0:train_len], examples[train_len:]

    def read_examples_inputs_arrays_without_occupations(self, examples):
        return list(map(lambda example: list(self._get_example_input_array_without_occupation(example)), examples))

    def read_examples_exists_outputs_arrays(self, examples):
        return list(map(lambda example: list(self._get_example_exists_output_array(example)), examples))

    def read_examples_net_outputs_arrays(self, examples):
        return list(map(lambda example: list(self._get_example_net_output_array(example)), examples))

    def _get_example_input_array_without_occupation(self, example):
        yield from self._generate_one_hot(example['input']['familyStatus'], self.FAMILY_STATUS_OPTIONS)
        yield from self._generate_one_hot(example['input']['educationType'], self.EDUCATION_TYPE_OPTIONS)
        yield from self._generate_one_hot(example['input']['jobSituation'], self.JOB_SITUATION_OPTIONS)
        yield from self._generate_one_hot(example['input']['industry'], self.INDUSTRY_OPTIONS)

        yield example['input']['benefitAgeLimit']
        yield example['input']['benefitAmount']
        yield example['input']['fractionOfficeWork']
        yield example['input']['staffResponsibility']
        yield example['input']['smoker']

        yield from self._generate_datetime_attributes(example['input']['birthday'])
        yield from self._generate_datetime_attributes(example['input']['insuranceStart'])

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
            example['output']['exists'],
            example['output']['net'],
            example['output']['gross']
        ]

    def _get_example_exists_output_array(self, example):
        return [
            example['output']['exists']
        ]

    def _get_example_net_output_array(self, example):
        return [
            example['output']['net']
        ]
