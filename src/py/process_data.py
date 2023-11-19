import yaml
import polars as pl
from math import log
from sklearn.model_selection import train_test_split


class PrepareData:
    def __init__(self) -> None:
        self.data_file = "data/data.csv"
        self.config_file = "src/yml/column_specifications.yml"

        self.test_size = 0.2

        self.is_transformed = False

        self.binary = False
        self.multiclass = False
        pass

    def load_data(self) -> None:
        """
        Loads the data specified in self.data_file, provided that it has been downloaded.
        """
        try:
            self.df = pl.read_csv(self.data_file)
        except FileNotFoundError:
            data_url = "https://github.com/velux-lead-data-scientist/velux_data_scientist/raw/main/data/data.csv"
            f"Please run `curl -LJO {data_url} data/data.csv` to download data"

        pass

    def load_config(self) -> None:
        """
        Loads the yaml file containing specifications on how to handle/aggregate the data input.
        """
        with open(self.config_file, "r") as file:
            self.config = yaml.safe_load(file)

        pass

    def select_transform_and_aggregate(self):
        """
        Subsets a dataframe to the set of columns specified in `config` and applies an
        x |--> log(x + 1) transform to the columns specified in `config` as well as binarizes
        any columns specified in `config`.
        Max-aggregates the transformed data to get bring the data down to unique IDs
        Input:
            - df: a polars dataframe
            - config: a dictionary
        Returns:
            - a polars dataframe
        """
        self.df = self.df[self.config.get("columns_to_keep", [])]

        self.df = self.df.select(
            pl.col(column).map_elements(lambda x: log(x + 1))
            if column in self.config.get("columns_to_log_transform", [])
            else column
            for column in self.df.columns
        )

        self.df = self.df.select(
            pl.col(column).map_elements(lambda x: int(x > 0))
            if column in self.config.get("columns_to_binarize", [])
            else column
            for column in self.df.columns
        )

        self.df = self.df.group_by("final_id").agg(
            pl.col(column).max() if column != "project_type" else pl.col(column).first()
            for column in self.df.columns
            if column != "final_id"
        )

        self.is_transformed = True

        pass

    def binary_dataset(self) -> None:
        """
        Constructs the dataset to be used for the binary classification task and splits
        it into a train and test set in a stratified manner
        """
        if not self.is_transformed:
            raise ValueError(
                "Please call `select_transform_and_aggregate` before calling `binary_dataset`"
            )

        if self.multiclass:
            raise ValueError(
                "Only one of `binary_dataset` and `multiclass_dataset` can be called per instance of `PrepareData`"
            )

        X, y = (
            self.df.drop(columns="project_type"),
            self.df["project_type"].map_elements(
                lambda x: 0 if x == "no_project_intent" else 1
            ),
        )

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y
        )

        pass

    def multiclass_dataset(self) -> None:
        """
        Constructs the dataset to be used for the multiclass classification task and splits
        it into a train and test set in a stratified manner
        """
        if not self.is_transformed:
            raise ValueError(
                "Please call `select_transform_and_aggregate` before calling `binary_dataset`"
            )

        if self.binary:
            raise ValueError(
                "Only one of `binary_dataset` and `multiclass_dataset` can be called per instance of `PrepareData`"
            )

        df_subset = self.df.filter(
            ~pl.col("project_type").is_in(["no_project_intent", "allprojects"])
        )

        renovation_types = ["atticrenovation", "loftconversion", "renovation"]
        construction_types = ["newbuild", "extension", "reroofing"]
        window_types = ["replacement", "upgrading"]

        X, y = (
            df_subset.drop(columns="project_type"),
            df_subset.with_columns(
                pl.when(pl.col("project_type").is_in(renovation_types))
                .then(pl.lit("renovation"))
                .when(pl.col("project_type").is_in(construction_types))
                .then(pl.lit("construction"))
                .when(pl.col("project_type").is_in(window_types))
                .then(pl.lit("windows"))
                .otherwise(pl.col("project_type"))
                .alias("y")
            )["y"],
        )

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y
        )

        pass

    def make_x_y(self, type: str) -> None:
        """
        Runs the sequence:
        load_data -> load_config -> select_transform_and_aggregate -> <binary/multyiclass>_dataset
        """
        self.load_data()
        self.load_config()
        self.select_transform_and_aggregate()
        if type.lower() == "binary":
            self.binary_dataset()
        elif type.lower() == "multiclass":
            self.multiclass_dataset()
        else:
            raise ValueError("`type` must be one of 'binary' or 'multiclass'")

        pass
