from src.data import DataProvider
from absl import logging
import os
import joblib
import pickle
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.utils import log_step
from src.config import INPUT_FILE, MODELS_DIR, TARGET_COL


@dataclass
class Predictor:

    model = pickle.load(open(os.path.join(MODELS_DIR, 'model.pkl'), 'rb'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'minmaxscaler.pkl'))

    def __init__(self, data, mode='predict'):
        self.data = data
        self.mode = mode

    def _preprocess_data(self):
        """Prepares prediction data for predictions

        :return: pd.DataFrame of data prepared for predictions
        """

        data_provider = DataProvider(
            data=self.data,
            mode=self.mode,
            scaler=self.scaler
        )
        self.x_sc = data_provider.prepare_prediction_data()
        return self.x_sc

    @log_step('Making predictions', logging.info)
    def _make_predictions(self):
        """Makes predictions

        :return: np.array of prediction values
        """

        self.predictions = self.model.predict(self.x_sc)
        return self.predictions

    @log_step('Unscaling predictions', logging.info)
    def _unscale_predictions(self):
        """Applies the inverse transformation of scaling

        :return: pd.DataFrame of unscaled predictions
        """

        df_predictions = pd.DataFrame(self.predictions, columns=[TARGET_COL])
        dummy_df = df_predictions.join(self.x_sc)
        unscaled_df = pd.DataFrame(self.scaler.inverse_transform(dummy_df), columns=dummy_df.columns)
        self.unscaled_predictions = unscaled_df[TARGET_COL]

    @log_step('Converting predicted values to integers', logging.info)
    def _round_predictions(self):
        """Rounds predicted values

        :return: pd.DataFrame of unscaled predictions rounded to integer values
        """

        self.unscaled_int_predictions = np.round(self.unscaled_predictions).astype(int)

    @log_step('Prediction phase', logging.info)
    def predict(self):
        """Function that:
                - prepares prediction data
                - makes predictions
                - unscales the predicted values
                - rounds the unscaled predicted values

        :return: pd.DataFrame of unscaled predicted values
        """

        self._preprocess_data()
        self._make_predictions()
        self._unscale_predictions()
        self._round_predictions()
        return self.unscaled_int_predictions


def main():
    logging.set_verbosity(logging.INFO)

    # since we haven't separated a part of the raw data for making predictions
    # we generate a dummy dataset based on the raw data by adding some noise to it
    dataset = pd.read_csv(INPUT_FILE)
    noise = np.random.normal(0, 1, [dataset.shape[0], dataset.shape[1]-3])
    non_float_cols = ['date', 'Appliances', 'lights']
    float_cols_with_noise = dataset.drop(non_float_cols, axis=1) + noise
    dummy_dataset = dataset[non_float_cols].join(float_cols_with_noise)

    # result
    predictor = Predictor(dummy_dataset)
    result = predictor.predict()
    print(result.head())


if __name__ == "__main__":
    main()
