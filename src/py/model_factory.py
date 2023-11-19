import importlib
import types

import yaml
from sklearn.metrics import get_scorer, make_scorer
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.multiclass import OneVsRestClassifier


def get_custom_scorer(scoring, **kwargs):
    """
    Sets up an sklearn scorer, either using a string to define a default scorer
    or using a dict to define a default scorer with custom arguments.
    """
    if isinstance(scoring, str):
        scorer = get_scorer(scoring)
        return make_scorer(scorer._score_func, **kwargs)
    else:
        return scoring


class ModelFactory:
    def __init__(self, config_file: str, multiclass=False):
        self.multiclass = multiclass
        self.config = self.get_config(config_file)
        self.model = self.register_model()
        self.type = self.config.get("type")
        self.param_grid = self.get_param_grid()
        self.register_logging_functions()
        self.scoring = self.register_metrics()
        self.refit = list(self.scoring.keys())[0]

    def get_config(self, config_file: str) -> dict:
        """
        Opens the specified configuration file.
        """
        with open(f"src/yml/{config_file}.yml", "r") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        return config

    def register_model(self):
        """
        Imports the model specified in the configuration file. In principle allows for
        custom models, but for now just uses standard sklearn models.

        In the case of multiclass data, the model is wrapped in OneVsRestClassifier. This
        can easily be modified to also allow models that are inherently multiclass by adding a
        tag to the yaml configs.
        """
        module_name, class_name = self.config.get("model").rsplit(".", 1)
        module = importlib.import_module(module_name)
        model = getattr(module, class_name)

        if self.multiclass and self.config.get("allow_multiclass", False):
            return OneVsRestClassifier(model())
        else:
            return model()

    def register_metrics(self):
        """
        Registers the relevant metrics to the model class based on the model type.
        Custom metrics are supplied as dicts and are handled separately from standard
        metrics, which are supplied as strings.
        """
        with open("src/yml/metrics.yml", "r") as metrics_file:
            metrics = yaml.safe_load(metrics_file)
        if self.multiclass:
            config_metrics = metrics.get("multiclass_metrics", [])
        else:
            config_metrics = metrics.get("classifier_metrics", [])

        metrics = dict()
        for metric in config_metrics:
            if isinstance(metric, str):
                metrics[metric] = get_scorer(metric)
            elif isinstance(metric, dict):
                name = metric["name"]
                cll = metric["callable"]
                kwargs = metric["kwargs"]
                metrics[name] = get_custom_scorer(cll, **kwargs)

        return metrics

    def get_param_grid(self):
        """
        Prepapres the parameter grid. If the models i wrapped in OVR, it needs an '__estimator__'
        prefix. If not, it just needs a '__' prefix to work with 'GridSearchCV'.
        """
        param_grid = self.config.get("parameters", {})

        if self.multiclass:
            param_grid = {
                "__estimator__" + str(key): val for key, val in param_grid.items()
            }
        else:
            param_grid = {"__" + str(key): val for key, val in param_grid.items()}

        return param_grid

    def register_logging_functions(self):
        """
        Registers the plot functions requested in the config file to the model class
        """
        logging_functions = self.config.get("logging_functions", [])
        logging_module = importlib.import_module("src.py.logging_functions")
        for log_function_name in logging_functions:
            setattr(
                self,
                log_function_name,
                types.MethodType(getattr(logging_module, log_function_name), self),
            )

        pass

    def call_log_functions(self, DataClass, logger):
        logging_functions = self.config.get("logging_functions", [])
        for log_function_name in logging_functions:
            getattr(self, log_function_name)(DataClass, logger)

        pass

    def grid_search(self, X, y, pipe, cv, n_jobs, verbose=0):
        """
        Helper function to perform a grid search to train the model and save relevant output.
        """
        GSCV = GridSearchCV(
            pipe,
            param_grid=self.param_grid,
            scoring=self.scoring,
            verbose=verbose,
            cv=cv,
            refit=self.refit,
            n_jobs=n_jobs,
        )
        GSCV.fit(X, y)
        self.best_estimator_ = GSCV.best_estimator_
        self.best_score_ = GSCV.best_score_
        self.cv_results_ = GSCV.cv_results_

        pass

    def make_cv(self, n_splits, n_repeats, seed):
        """
        Sets up a crossvalidation scheme to be used in, e.g., `grid_search`
        """
        self.cv = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=seed
        )

        pass
