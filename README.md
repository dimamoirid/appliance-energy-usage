# appliance-energy-usage 

This is an example project for demonstrating the quality of the 
code I am able to develop. Achieving a decent performance for the 
model is beyond the scope of this project.

Important notes:
- An elementary EDA has been performed and it's not included here
- Frequently used constant values are stored in the _config.py_ file
- A logging decorator is stored separately in the _utils.py_ file
- The code is modularized, therefore the `pip install -e .` command
should be run from a terminal (see _Setup_ below)

#### Data
UCI Machine Learning Repository: Appliances energy prediction Data Set 
(https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction)

If for any reason the data is not available, one must download the 
_energy_data_complete_ csv file from the _Data Folder_ of the link 
above and store it into the project's _data_ folder in order to be able 
to reproduce the results.

#### Model
A RandomForest regressor from the scikit-learn library is used.
More naive (e.g. Linear or Ridge) or more powerful (e.g. XGBoost 
or Neural Network) regressors could have been trained, but as 
explained above this is not the point of this project.

# Setup
```
git clone https://github.com/dimamoirid/appliance-energy-usage.git
cd appliance-energy-usage

conda env create -f environment.yml
conda activate appliance-energy-usage

pip install -e .
```

# Usage
1. Training: `python src/train.py`
2. Evaluation: `python src/eval.py`
3. Prediction: `python src/predict.py`

#### Outputs
If step 1 is run properly, a ___model.pkl___ as well as a 
___minmaxscaler.pkl___ file should be created and saved inside 
the _models_ folder. By running the second command a 
___metrics.txt___ file is created and saved inside the _reports_ 
folder and the output of the third step is displayed in the 
terminal where the ___first five predictions___ are printed.
