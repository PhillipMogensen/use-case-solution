import os
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
            raise FileNotFoundError(
                f"Please run `curl -LJO {data_url}` from inside data/"
            )

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
        if os.path.exists("data/binary_train.csv") or os.path.exists(
            "data/binary_test.csv"
        ):
            raise Warning("`binary_dataset` has already been run")

        if not self.is_transformed:
            raise ValueError(
                "Please call `select_transform_and_aggregate` before calling `binary_dataset`"
            )

        if self.multiclass:
            raise ValueError(
                "Only one of `binary_dataset` and `multiclass_dataset` can be called per instance of `PrepareData`"
            )

        self.df = self.df.with_columns(
            pl.when(pl.col("project_type") == "no_project_intent")
            .then(pl.lit(0))
            .otherwise(pl.lit(1))
            .alias("project_type")
        )

        df_train, df_test = train_test_split(
            self.df,
            test_size=self.test_size,
            stratify=self.df["project_type"],
            random_state=1234,
        )

        df_train.write_csv("data/binary_train.csv")
        df_test.write_csv("data/binary_test.csv")

        pass

    def multiclass_dataset(self) -> None:
        """
        Constructs the dataset to be used for the multiclass classification task and splits
        it into a train and test set in a stratified manner
        """
        if os.path.exists("data/multiclass_train.csv") or os.path.exists(
            "data/multiclass_test.csv"
        ):
            raise Warning("`binary_dataset` has already been run")

        if not self.is_transformed:
            raise ValueError(
                "Please call `select_transform_and_aggregate` before calling `binary_dataset`"
            )

        if self.binary:
            raise ValueError(
                "Only one of `binary_dataset` and `multiclass_dataset` can be called per instance of `PrepareData`"
            )

        renovation_types = ["atticrenovation", "loftconversion", "renovation"]
        construction_types = ["newbuild", "extension", "reroofing"]
        window_types = ["replacement", "upgrading"]

        df_subset = self.df.filter(
            ~pl.col("project_type").is_in(["no_project_intent", "allprojects"])
        ).with_columns(
            pl.when(pl.col("project_type").is_in(renovation_types))
            .then(pl.lit("renovation"))
            .when(pl.col("project_type").is_in(construction_types))
            .then(pl.lit("construction"))
            .when(pl.col("project_type").is_in(window_types))
            .then(pl.lit("windows"))
            .otherwise(pl.col("project_type"))
            .alias("project_type")
        )

        df_train, df_test = train_test_split(
            df_subset,
            test_size=self.test_size,
            stratify=df_subset["project_type"],
            random_state=1234,
        )

        df_train.write_csv("data/multiclass_train.csv")
        df_test.write_csv("data/multiclass_test.csv")

        pass

    def make_x_y(self, type: str) -> None:
        """
        Runs the sequence:
        load_data -> load_config -> select_transform_and_aggregate -> <binary/multyiclass>_dataset
        """
        if type.lower() == "binary":
            files = ["data/binary_train.csv", "data/binary_test.csv"]
            make_data = self.binary_dataset
        elif type.lower() == "multiclass":
            files = ["data/multiclass_train.csv", "data/multiclass_test.csv"]
            make_data = self.multiclass_dataset
        else:
            raise ValueError("`type` must be one of 'binary' or 'multiclass'")

        if not (os.path.exists(files[0]) or os.path.exists(files[1])):
            self.load_data()
            self.load_config()
            self.select_transform_and_aggregate()
            make_data()

        df_train = pl.read_csv(files[0])
        df_test = pl.read_csv(files[1])

        self.X_train, self.y_train = (
            df_train.drop(columns=["project_type", "final_id"]),
            df_train["project_type"],
        )
        self.X_test, self.y_test = (
            df_test.drop(columns=["project_type", "final_id"]),
            df_test["project_type"],
        )

        pass
