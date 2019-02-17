import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
from sources.DataProvider import DataProvider
import autosklearn.regression

from keras.callbacks import TensorBoard
import time

def main():


    data_provider = DataProvider()
    X, y = data_provider.load_from_file()

    feature_types = (['categorical'] * 26) + (['numerical'] * 4) + ['categorical'] + (['numerical'] * 10)
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)

    
    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=120,
        per_run_time_limit=30,
        tmp_folder='tmp/autosklearn_regression_example_tmp2',
        output_folder='tmp/autosklearn_regression_example_out2',
    )
    automl.fit(X_train, y_train, dataset_name='set_1151',
               feat_type=feature_types)

    print(automl.show_models())
    predictions = automl.predict(X_test)
    print("mean_absolute_error:", sklearn.metrics.mean_absolute_error(y_test, predictions))
    print("median_absolute_error:", sklearn.metrics.median_absolute_error(y_test, predictions))
    print("explained_variance_score:", sklearn.metrics.explained_variance_score(y_test, predictions))

if __name__ == '__main__':
    main()