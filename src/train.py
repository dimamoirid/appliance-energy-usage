from absl import flags, logging
import pandas as pd
from pathlib import Path
import sys
import os
import pickle
from sklearn.ensemble import RandomForestRegressor

from src.data import DataProvider
from src.config import INPUT_FILE, MODELS_DIR


def main():
    logging.set_verbosity(logging.INFO)

    flags.DEFINE_string('input_file', INPUT_FILE, 'File containing the raw data')
    flags.DEFINE_string('mode', 'train', 'Either train or predict')
    flags.DEFINE_string('model_file_dir', os.path.join(MODELS_DIR, 'model.pkl'), 'Directory containing fitted model')

    FLAGS = flags.FLAGS

    if Path(sys.argv[0]).name == 'pydevconsole.py':
        # interactive mode
        FLAGS([__name__])
    else:
        # command-line mode
        FLAGS(sys.argv)

    # get data
    data_provider = DataProvider(
        data=pd.read_csv(FLAGS.input_file),
        mode=FLAGS.mode,
    )
    train_sc_x, train_sc_y = data_provider.prepare_training_data()

    # train model
    model = RandomForestRegressor(max_depth=5, random_state=42)
    model.fit(train_sc_x, train_sc_y)

    # save model
    pickle.dump(model, open(FLAGS.model_file_dir, 'wb'))


if __name__ == '__main__':
    main()
