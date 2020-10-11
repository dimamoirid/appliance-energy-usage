from absl import logging
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import joblib
import pandas as pd

from src.utils import log_step
from src.config import MODELS_DIR, TARGET_COL, REDUNDANT_FEATURES


@dataclass
class DataProvider:

    def __init__(self, data, mode, scaler=None):
        self.data = data
        self.mode = mode
        self.scaler = scaler

    @log_step('Removing redundant features', logging.info)
    def _drop_redundant_features(self, redundant_features):
        """Removes the redundant features

        :param redundant_feature: List of redundant features
        :return: Input df without without the redundant features
        """

        self.data = self.data.drop(redundant_features, axis=1)
        return self.data

    @log_step('Splitting into train & test', logging.info)
    def _split_data(self):
        """Splits data into train and test set

        :return: np.arrays of the train and test sets
        """

        self.train, self.test = train_test_split(self.data, test_size=0.25, random_state=42)
        return self.train, self.test

    @log_step('Scaling data', logging.info)
    def _scale_data(self):
        """Applies scaling

        :return:
            scaled train and test sets if mode=train
            scaled train and test sets if mode=eval
            scaled data if mode=predict
        """

        if self.mode == 'train':
            scaler = MinMaxScaler().fit(self.train)
            joblib.dump(scaler, os.path.join(MODELS_DIR, 'minmaxscaler.pkl'))
            self.train_sc = pd.DataFrame(scaler.transform(self.train), columns=self.data.columns)
            self.test_sc = pd.DataFrame(scaler.transform(self.test), columns=self.data.columns)
            return self.train_sc, self.test_sc

        elif self.mode == 'eval':
            # load pretrained scaler and scale only test data
            scaler = self.scaler
            self.train_sc = pd.DataFrame(scaler.transform(self.train), columns=self.data.columns)
            self.test_sc = pd.DataFrame(scaler.transform(self.test), columns=self.data.columns)
            return self.train_sc, self.test_sc

        elif self.mode == 'predict':
            # load pretrained scaler and scale prediction data
            scaler = self.scaler
            self.data_sc = pd.DataFrame(scaler.transform(self.data), columns=self.data.columns)
            return self.data_sc

        else:
            raise ValueError("Mode should be equal to 'train', 'eval' or 'predict'")

    @log_step('Separating independent from dependent variables', logging.info)
    def _divide_into_x_and_y(self):
        """Divides into independent and dependent variables sets

        :return:
            independent and dependent train sets if mode=train
            independent and dependent train and test sets if mode=eval
            independent set of variables if mode=predict
        """

        if self.mode == 'train':
            self.train_sc_x = self.train_sc.drop([TARGET_COL], axis=1)
            self.train_sc_y = self.train_sc[TARGET_COL]
            return self.train_sc_x, self.train_sc_y

        elif self.mode == 'eval':
            self.train_sc_x = self.train_sc.drop([TARGET_COL], axis=1)
            self.train_sc_y = self.train_sc[TARGET_COL]
            self.test_sc_x = self.test_sc.drop([TARGET_COL], axis=1)
            self.test_sc_y = self.test_sc[TARGET_COL]
            return self.train_sc_x, self.train_sc_y, self.test_sc_x, self.test_sc_y

        elif self.mode == 'predict':
            self.x_sc = self.data_sc.drop([TARGET_COL], axis=1)
            return self.x_sc

        else:
            raise ValueError("Mode should be equal to 'train', 'eval' or 'predict'")

    @log_step('Preparing data for training', logging.info)
    def prepare_training_data(self):
        """Function that:
                - removes the redundant features from the raw data
                - splits the data into train and test
                - scales the train and test data
                - divides data into independent and dependent variables sets

        :return: Scaled independent and dependent variables train sets
        """

        self._drop_redundant_features(REDUNDANT_FEATURES)
        self._split_data()
        self._scale_data()
        self._divide_into_x_and_y()
        return self.train_sc_x, self.train_sc_y

    @log_step('Preparing data for evaluation', logging.info)
    def prepare_evaluation_data(self):
        """Function that:
                - removes the redundant features from the raw data
                - splits the data into train and test
                - scales the train and test data
                - divides data into independent and dependent variables sets

        :return: Scaled independent and dependent variables train and test sets
        """

        self._drop_redundant_features(REDUNDANT_FEATURES)
        self._split_data()
        self._scale_data()
        self._divide_into_x_and_y()
        return self.train_sc_x, self.train_sc_y, self.test_sc_x, self.test_sc_y

    @log_step('Preparing prediction data', logging.info)
    def prepare_prediction_data(self):
        """Function that:
                - removes the redundant features from the raw data
                - scales the data
                - divides data into independent and dependent variables sets

        :return: Scaled independent variables set
        """

        self._drop_redundant_features(REDUNDANT_FEATURES)
        self._scale_data()
        self._divide_into_x_and_y()
        return self.x_sc
