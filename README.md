# Downloading data
To download the data file the project, run the below command from inside the `data/` directory:
```
curl -LJO "https://github.com/velux-lead-data-scientist/velux_data_scientist/raw/main/data/data.csv"
```

# Project dependencies
To run the Quarto notebook, which contains only `R`, a valid `R` (>4.2.3) installation with `renv` installed is needed. The `R` environment is managed by `renv` and can be reproduced by running `renv::restore()`. Note that it is not necessary to run the Quarto notebook. A pre-rendered `.html` is available at `notebooks/explore.html`. This notebook contains the general considerations I made when having a first look at the data.

To run the python part of the project a valid `Python` (>3.12) and `poetry` is needed. In addition, a valid ClearML configuration file must exist in the local environment. The environment is managed by `poetry`. 

# Data overview
Per the observations in `explore.html`, we split the raw `data/data.csv` into two separate datasets: one for classifying project intent from no project intent (referred to as the `binary` dataset) and one for classifying project types from one another among customers/sessions with known/assumed intent (referred to as the `multiclass` dataset).

# Training and testing models
The project contains two main files: `train.py` and `test.py`. To train model, you can use
```
python train.py <dataset> <model> --<extra_arguments>
```
Run
```
python train.py --help
```
to see a list of available commands. `train.py` trains a specified model, records various metrics and records the run on ClearML (self-hosted, not publicly available). `test.py` takes in a `run_id` from ClearML and applies the trained model to the test data.

Under the hood are various python functions and classes in `src/py`, chief among which are:

-   `src.py.process_data.PrepareData`: a class which takes in the raw `data/data.csv`, subsets and transforms the columns and rows (per the observations in `explore.html`), converts into two datasets (binary/multiclass) and splits both into train and test datasets, which are then stored in `data/`.
-   `src.py.model_factory.ModelFactory`: a class which takes in a model configuration file and a metric configuration file (stored in `src/yml`), loads the specified model, registers the metrics and sets up a `GridSearchCV` instance and a crossvalidation scheme.

In `src.py.logging_functions` are various functions that gets dymanically added to an instance of `ModelFactory` depending on what is specified in the configuration file. These functions are helper functions to record figures for, e.g., the permutation imortance.