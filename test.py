from clearml import Task
import joblib
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, roc_curve, auc
import matplotlib.pyplot as plt
import argparse
from sklearn.preprocessing import label_binarize
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Test a trained model from ClearML")
    parser.add_argument(
        "run_id",
        type=str,
        help="The run_id from ClearML",
    )

    return parser.parse_args()


def multiclass_roc(model, class_names, n_classes, y_test, y_score):
    fpr, tpr, roc_auc = dict(), dict(), dict()

    fig, ax = plt.subplots(dpi=300)

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr_grid = np.linspace(0.0, 1.0, 1000)

    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)

    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

    # Average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    ax.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    ax.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colormap = plt.get_cmap("Paired", n_classes)
    colors = [colormap(i) for i in range(n_classes)]
    for class_name, color in zip(class_names, colors):
        i = [i for i, c in enumerate(class_names) if c == class_name]
        RocCurveDisplay.from_predictions(
            y_test[:, i],
            y_score[:, i],
            name=f"ROC curve for {class_name}",
            color=color,
            ax=ax,
            plot_chance_level=(class_name == class_names[-1:][0]),
        )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve per class and averages")
    ax.legend()

    return fig


def test_run(run_id):
    """
    Fetches a trained model from ClearML and outputs ROC/PR curves. Does not output a multiclass
    PR curve.
    """
    task = Task.get_task(task_id=run_id)

    model_artifact = task.artifacts["model"].get_local_copy()
    X_artifact = task.artifacts["X_test"].get_local_copy()
    y_artifact = task.artifacts["y_test"].get_local_copy()
    model = joblib.load(model_artifact)
    X_test = joblib.load(X_artifact).to_pandas()
    y_test = joblib.load(y_artifact).to_pandas()

    class_names = model.classes_
    n_classes = len(class_names)

    if n_classes <= 2:
        fig_roc, ax_roc = plt.subplots(dpi=300)
        RocCurveDisplay.from_estimator(
            model, X_test, y_test, ax=ax_roc, plot_chance_level=True
        )

        fig_pr, ax_pr = plt.subplots(dpi=300)
        PrecisionRecallDisplay.from_estimator(
            model, X_test, y_test, ax=ax_pr, plot_chance_level=True
        )

        fig_pr.savefig(f"PR_{run_id}.png")
    else:
        y_test_ = label_binarize(y_test, classes=class_names)
        y_score = model.predict_proba(X_test)

        fig_roc = multiclass_roc(model, class_names, n_classes, y_test_, y_score)

        fig_pr, ax_pr = plt.subplots()

    fig_roc.savefig(f"ROC_{run_id}.png")

    pass


if __name__ == "__main__":
    args = parse_args()
    test_run(args.run_id)
