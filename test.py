from clearml import Task
import joblib
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
import matplotlib.pyplot as plt
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Test a trained model from ClearML")
    parser.add_argument(
        "run_id",
        type=str,
        help="The run_id from ClearML",
    )

    return parser.parse_args()


def test_run(run_id):
    """
    Fetches a trained model from ClearML and outputs ROC/PR curves
    """
    task = Task.get_task(task_id=run_id)

    model_artifact = task.artifacts["model"].get_local_copy()
    X_artifact = task.artifacts["X_test"].get_local_copy()
    y_artifact = task.artifacts["y_test"].get_local_copy()
    model = joblib.load(model_artifact)
    X_test = joblib.load(X_artifact).to_pandas()
    y_test = joblib.load(y_artifact).to_pandas()

    fig_roc, ax = plt.subplots(dpi=300)
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, plot_chance_level=True)

    fig_roc.savefig(f"ROC_{run_id}.png")

    fig_pr, ax = plt.subplots(dpi=300)
    PrecisionRecallDisplay.from_estimator(
        model, X_test, y_test, ax=ax, plot_chance_level=True
    )

    fig_pr.savefig(f"PR_{run_id}.png")

    pass


if __name__ == "__main__":
    args = parse_args()
    test_run(args.run_id)
