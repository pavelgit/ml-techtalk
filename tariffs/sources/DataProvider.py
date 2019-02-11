from pymongo import MongoClient
import numpy as np
import pickle
import math 


class DataProvider:

    NO_OCCUPATION_INPUT_ROW_LENGTH = 33

    def load_from_db(self):
        with MongoClient('localhost', 27017) as client:
            db = client['bu-config']
            examples_collection = db['fb_1151']
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

    def read_examples_inputs_arrays_without_occupations(self, examples):
        return list(map(lambda example: self._get_example_input_array_without_occupation(example), examples))

    def read_examples_exists_outputs_arrays(self, examples):
        return list(map(lambda example: self._get_example_exists_output_array(example), examples))

    def _get_example_input_array_without_occupation(self, example):
        return [
            example['input']['family']['noInput'],
            example['input']['family']['pairWithChildren'],
            example['input']['family']['singleWithChildren'],
            example['input']['family']['noChildren'],

            example['input']['birthday'],

            example['input']['educationType']['promotion'],
            example['input']['educationType']['studiumMasterOderDiplom'],
            example['input']['educationType']['studiumBachelor'],
            example['input']['educationType']['meister'],
            example['input']['educationType']['berufsausbildung'],
            example['input']['educationType']['abitur'],
            example['input']['educationType']['realschulabschluss'],
            example['input']['educationType']['hauptschulabschluss'],
            example['input']['educationType']['keinAbschluss'],


            example['input']['jobSituation']['angestelltSelbststaendig'],
            example['input']['jobSituation']['beamter'],
            example['input']['jobSituation']['sonstiges'],

            example['input']['benefitAgeLimit'],
            example['input']['benefitAmount'],
            example['input']['fractionOfficeWork'],

            example['input']['industry']['elektro'],
            example['input']['industry']['gesundheitswesen'],
            example['input']['industry']['holz'],
            example['input']['industry']['it'],
            example['input']['industry']['kunststoff'],
            example['input']['industry']['metall'],
            example['input']['industry']['stahl'],
            example['input']['industry']['textil'],
            example['input']['industry']['sonstigeBranche'],

            example['input']['staffResponsibility'],
            example['input']['smoker'],
            example['input']['salutation']['herr'],
            example['input']['salutation']['frau'],
        ]

    def _get_example_output_array(self, example):
        return [
            example['result']['exists'],
            example['result']['netPrice'],
            example['result']['grossPrice']
        ]

    def _get_example_exists_output_array(self, example):
        return [
            example['result']['exists']
        ]
