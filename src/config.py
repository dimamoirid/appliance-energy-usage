import os

ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'data')
SRC_DIR = os.path.join(ROOT_DIR, 'src')

INPUT_FILE = os.path.join(DATA_DIR, 'energydata_complete.csv')
REPORTS_DIR = os.path.join(ROOT_DIR, 'reports')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
REDUNDANT_FEATURES = ['date', 'lights', 'rv1', 'rv2', 'Visibility', 'T6', 'T9']
TARGET_COL = 'Appliances'
