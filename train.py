import time
import joblib
from sklearn.decomposition import PCA
from src.py.model_factory import ModelFactory
from sklearn.preprocessing import StandardScaler
from src.py.process_data import PrepareData
from clearml import Task
from sklearn.pipeline import Pipeline
import hashlib
import argparse
from sklearn import set_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a specified model on a specified dataset"
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="The dataset to train on. Can be 'binary' or 'multiclass'",
    )
    parser.add_argument(
        "model",
        type=str,
        help="The model to train. Must be the name of a model .yml file in src/yml/",
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        required=False,
        default=None,
        help="Adds tags to the ClearML task",
    )
    parser.add_argument(
        "--folds",
        type=int,
        required=False,
        default=5,
        help="Number of folds to use in the crossvalidation scheme",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        required=False,
        default=10,
        help="Number of repeats to use in the crossvalidation scheme",
    )
    parser.add_argument(
        "--pca",
        type=int,
        default=None,
        help="Number of pca components to include, if doing pca for dim reduction. Use None to omit.",
    )
    parser.add_argument(
        "--scale",
        action=argparse.BooleanOptionalAction,
        help="Flag to add scaling to the pipeline",
    )
    parser.add_argument(
        "--select",
        type=str,
        nargs="+",
        required=False,
        default=None,
        help="columns to subset to, if so desired",
    )

    return parser.parse_args()


def hash_args(args: argparse) -> str:
    """
    Helper function to convert the supplied command line arguments to a unique ID.
    """
    # Get all arguments
    args_dict = vars(args)

    # If the contents of an argument is a list, we sort that first:
    for key, item in args_dict.items():
        if isinstance(item, list):
            args_dict[key] = sorted(item)

    sorted_items = sorted(args_dict.items(), key=lambda x: (x[0], x[1]))
    args_string = str(sorted_items)
    unique_id = hashlib.md5(args_string.encode()).hexdigest()

    return unique_id


def make_pipe(args: argparse, model) -> Pipeline:
    """
    Helper function to set up a model pipeline
    """
    pipe = Pipeline([("", model)])
    if args.pca:
        pipe.steps.insert(0, ("pca", PCA(n_components=args.pca)))
    if args.scale:
        pipe.steps.insert(0, ("scaler", StandardScaler()))

    return pipe


def train():
    """
    Trains a model as specified by command-line arguments and logs to ClearML
    """
    set_config(transform_output="pandas")
    args = parse_args()

    task = Task.init(
        project_name=args.dataset,
        task_name=f"{args.model}_{hash_args(args)}",
        tags=args.tags,
        auto_resource_monitoring=False,
    )
    logger = task.get_logger()

    if args.dataset.lower() == "binary":
        multiclass = False
    elif args.dataset.lower() == "multiclass":
        multiclass = True
    else:
        raise ValueError("argument 'dataset' must be one of 'binary' or 'multiclass'")

    dataclass = PrepareData()
    dataclass.make_x_y(args.dataset)

    if args.select:
        dataclass.X_train = dataclass.X_train[args.select]
        dataclass.X_test = dataclass.X_test[args.select]

    mf = ModelFactory(args.model, multiclass)

    print("Starting crossvalidation")
    mf.make_cv(n_splits=args.folds, n_repeats=args.repeats, seed=1)
    pipe = make_pipe(args, mf.model)
    t0 = time.time()
    mf.grid_search(
        X=dataclass.X_train,
        y=dataclass.y_train,
        pipe=pipe,
        cv=mf.cv,
        n_jobs=-1,
    )
    t0 = time.time() - t0
    print(f"Crossvalidation finished in {t0:.2f} seconds")

    # Log all functions attached via the models config file.
    mf.call_log_functions(dataclass, logger)

    # Log best found model:
    joblib.dump(
        mf.best_estimator_, f"{args.dataset}_model_{hash_args(args)}.pkl", compress=True
    )
    task.upload_artifact(
        "model", artifact_object=f"{args.dataset}_model_{hash_args(args)}.pkl"
    )
    task.upload_artifact("X_test", artifact_object=dataclass.X_test)
    task.upload_artifact("y_test", artifact_object=dataclass.y_test)

    # Log best score
    logger.report_scalar(
        title="Grid Search Results",
        series="Best Score",
        value=mf.best_score_,
        iteration=1,
    )

    print(f"Best achieved score: {mf.best_score_:.4f}")

    task.close()

    pass


if __name__ == "__main__":
    train()
