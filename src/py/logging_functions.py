import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
from src.py.utils import timeit
from sklearn.inspection import permutation_importance
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from sklearn.base import clone
from scipy.signal import savgol_filter

from src.py.model_factory import ModelFactory


@timeit("computation of feature importances")
def feature_importances(self: ModelFactory, DataClass, logger):
    """
    Input:
        - `self`: Will be assigned a ModelFactory when called in `ModelFactory`
        - `DataClass`: An ML class from an ML component.
        - `logger`: A ClearML logging object
    Objects to be logged to ClearML / to be returned:
        - A figure of permutation importances based on the chosen (in ModelFactory) metric
        will be logged to ClearML. If `logger` is None, the figure is returned instead.
    """
    result = permutation_importance(
        self.best_estimator_,
        DataClass.X_train.to_pandas(),
        DataClass.y_train,
        n_repeats=10,
        n_jobs=-1,
        scoring=self.refit,
    )
    sorted_indices = sorted(
        range(len(result.importances_mean)),
        key=lambda k: result.importances_mean[k],
        reverse=True,
    )
    importances = pd.Series(
        [result.importances_mean[i] for i in sorted_indices],
        index=DataClass.X_train.to_pandas().columns[sorted_indices],
    )
    fig, ax = plt.subplots()
    importances.plot.bar(
        yerr=[result.importances_std[i] for i in sorted_indices], ax=ax
    )
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel(f"Mean {self.refit} decrease")
    fig.tight_layout()

    if logger is not None:
        logger.report_matplotlib_figure(
            title="Permutation importance", series="", figure=fig
        )

        pass
    else:
        return fig


@timeit("computation of ROC/PR curves")
def cv_curves(self: ModelFactory, DataClass, logger):
    """
    Input:
        - `self`: Will be assigned a ModelFactory when called in `ModelFactory`
        - `DataClass`: An ML class from an ML component.
        - `logger`: A ClearML logging object
    Objects to be logged to ClearML:
        - A figure of the ROC curves as achieved on each fold in the gridsearch
        - A figure of the precision recall curves as achieved on each fold in the gridsearch
    """
    class_names = self.best_estimator_.classes_
    n_classes = len(class_names)
    colormap = plt.get_cmap("Paired", n_classes)

    def plot_class_curves(ax_roc, ax_pr, class_i):
        roc_data = []
        pr_data = []

        for i, (train, test) in enumerate(
            self.cv.split(DataClass.X_train, DataClass.y_train)
        ):
            X_train = DataClass.X_train.to_pandas().iloc[train]
            X_test = DataClass.X_train.to_pandas().iloc[test]
            y_train = DataClass.y_train.to_pandas().iloc[train]
            y_test = DataClass.y_train.to_pandas().iloc[test]

            y_test = (y_test == class_names[class_i]).astype(int)

            model = clone(self.best_estimator_)
            model.fit(X_train, y_train)
            y_score = model.predict_proba(X_test)[:, class_i]

            roc = RocCurveDisplay.from_predictions(
                y_test, y_score, ax=ax_roc, alpha=0.1, color=colormap(class_i)
            )
            pr = PrecisionRecallDisplay.from_predictions(
                y_test, y_score, ax=ax_pr, alpha=0.1, color=colormap(class_i)
            )

            roc_data.append(
                pd.DataFrame(roc.line_.get_xydata(), columns=["fpr", "tpr"])
            )
            pr_data.append(
                pd.DataFrame(pr.line_.get_xydata(), columns=["recall", "precision"])
            )

        roc_data = pd.concat(roc_data).groupby("fpr", as_index=False).agg("mean")
        pr_data = pd.concat(pr_data).groupby("recall", as_index=False).agg("mean")

        roc_smooth = savgol_filter(roc_data["tpr"], 5, 3)
        ax_roc.plot(roc_data["fpr"], roc_smooth, color=colormap(class_i))

        pr_smooth = savgol_filter(pr_data["precision"], 5, 3)
        ax_pr.plot(pr_data["recall"], pr_smooth, color=colormap(class_i))

        pass

    if n_classes <= 2:
        fig_roc, ax_roc = plt.subplots(dpi=300)
        fig_pr, ax_pr = plt.subplots(dpi=300)

        plot_class_curves(ax_roc, ax_pr, class_i=1)

        ax_roc.get_legend().remove()
        ax_pr.get_legend().remove()

        base_precision = DataClass.y_train.mean()
        ax_pr.plot([0, 1], [base_precision, base_precision], "--")
    else:
        fig_roc, ax_roc = plt.subplots(dpi=300)
        fig_pr, ax_pr = plt.subplots(dpi=300)
        dummy_lines = []
        for i, class_name in enumerate(class_names):
            plot_class_curves(ax_roc, ax_pr, class_i=i)
            dummy_lines.append(
                mlines.Line2D([], [], color=colormap(i), label=f"Class {class_name}")
            )

        ax_roc.get_legend().remove()
        ax_pr.get_legend().remove()

        ax_roc.legend(handles=dummy_lines)
        ax_pr.legend(handles=dummy_lines)

        for i, class_name in enumerate(class_names):
            base_precision = DataClass.y_train.map_elements(
                lambda x: 1 if x == class_name else 0
            ).mean()
            ax_pr.plot(
                [0, 1], [base_precision, base_precision], "--", color=colormap(i)
            )

    ax_roc.plot([0, 1], [0, 1], "--")

    logger.report_matplotlib_figure(
        title="Cross validated ROC curve", series="", figure=fig_roc, report_image=True
    )

    logger.report_matplotlib_figure(
        title="Cross validated precision recall curve",
        series="",
        figure=fig_pr,
        report_image=True,
    )
