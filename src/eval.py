import pandas as pd
from absl import flags, logging
import sys
import os
import pickle
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data import DataProvider
from src.config import INPUT_FILE, REPORTS_DIR, MODELS_DIR


def main():
    logging.set_verbosity(logging.INFO)

    flags.DEFINE_string('input_file', INPUT_FILE, 'File containing the raw data')
    flags.DEFINE_string('reports_dir', REPORTS_DIR, 'Directory containing reports')
    flags.DEFINE_string('mode', 'eval', 'Either train or predict')
    flags.DEFINE_string('model_file', os.path.join(MODELS_DIR, 'model.pkl'), 'File containing fitted model')
    flags.DEFINE_string('scaler_file', os.path.join(MODELS_DIR, 'minmaxscaler.pkl'), 'File containing the scaler')

    FLAGS = flags.FLAGS

    if Path(sys.argv[0]).name == 'pydevconsole.py':
        # interactive mode
        FLAGS([__name__])
    else:
        # command-line mode
        FLAGS(sys.argv)

    # load model
    model = pickle.load(open(FLAGS.model_file, 'rb'))

    # get data
    data_provider = DataProvider(
        data=pd.read_csv(FLAGS.input_file),
        mode=FLAGS.mode,
        scaler=joblib.load(FLAGS.scaler_file)
    )
    train_sc_x, train_sc_y, test_sc_x, test_sc_y = data_provider.prepare_evaluation_data()

    # make predictions on train (train_sc_x) and test (test_sc_x) scaled data
    predictions_train = model.predict(train_sc_x)
    predictions_test = model.predict(test_sc_x)

    # compare predictions with actual data
    metrics = {
        "mae": mean_absolute_error,
        "mse": mean_squared_error,
    }

    f = open(os.path.join(FLAGS.reports_dir, 'metrics.txt'), 'w+')
    for name, function in metrics.items():
        train_error = function(train_sc_y, predictions_train)
        test_error = function(test_sc_y, predictions_test)
        f.write(f"Train {name} = %f - Test {name} = %f \n" % (train_error, test_error))
    f.close()


if __name__ == '__main__':
    main()
