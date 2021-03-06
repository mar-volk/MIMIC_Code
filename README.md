# MIMIC_Code
In this repo I present how I
- access data from the MIMIC III database
- how I perform classification and regressions with it

## Setup

Clone the repo from 
https://github.com/mar-volk/MIMIC_Code .


```sh
conda env create --file environment.yml
source activate MIMIC_Code
```

Request access for the MIMIC III database as explained on
https://mimic.physionet.org

Copy unzipped MIMIC-data into
'data/all/raw'

Copy unzipped MIMIC-demo-data (smaller files for testing) into 
'data/demo_data/all/raw'


## Usage

```sh
conda env update --file environment.yml
source activate MIMIC_Code
jupyter notebook
pytest
```

Run the following script to split the data into training, validation and test data
```sh
src/prep/train_test_split.py
```

Click through following jupyter notebook to get an overview:
exp/Examples_on_how_to_prepare_predictors_and_targets.ipynb
exp/Evaluate_NN_Classifier_07.ipynb


