from abc import ABC, abstractmethod
from typing import Optional, List, Union
import numpy as np
import pandas as pd
import time

from setar_trees.utils import (
    find_cut_point,
    fit_global_model,
    create_split,
    SS,
    get_leaf_index,
    check_linearity,
    check_error_improvement,
)

LINEAR_STOPPING_CRITERION = "lin_test"
ERROR_STOPPING_CRITERION = "error_imp"
HYBRID_STOPPING_CRITERION = "both"
STOPPING_CRITERIA_LIST = [
    LINEAR_STOPPING_CRITERION,
    ERROR_STOPPING_CRITERION,
    HYBRID_STOPPING_CRITERION,
]


# Define an abstract base class for all SETAR models
class BaseSETAR(ABC):
    def __init__(
        self,
        lag: int,
        forecast_horizon: int,
        exogenous_covariates: List[str] = None,
        depth: int = 8,
        significance: float = 0.05,
        min_sample_split: int = 5,
        fixed_lag: bool = False,
        external_lag: int = 1,
        stopping_criteria: str = "both",
        error_threshold: float = 0.05,
        seq_significance: bool = True,  # whether to apply significance_divider
        significance_divider: float = 2,
        verbose: int = 0,
    ):
        for name, var in zip(
            [
                "lag",
                "forecast_horizon",
                "depth",
                "min_sample_split",
                "external_lag",
                "significance_divider",
            ],
            [
                lag,
                forecast_horizon,
                depth,
                min_sample_split,
                external_lag,
                significance_divider,
            ],
        ):
            assert var >= 1, f"min `{name}` value is 1, value is {var}"

        for name, var in zip(
            ["significance", "error_threshold"],
            [significance, error_threshold],
        ):
            assert var > 0 and var < 1, f"`{name}` must be in (0,1)"

        assert (
            stopping_criteria in STOPPING_CRITERIA_LIST
        ), f"""`stopping_criteria` must be in {STOPPING_CRITERIA_LIST},
             value was {stopping_criteria}"""

        self.lag = lag
        self.forecast_horizon = forecast_horizon
        self.exogenous_covariates = exogenous_covariates
        self.depth = depth
        self.significance = significance

        sample_split_lower_bound = (
            lag
            if exogenous_covariates is None
            else lag + len(exogenous_covariates)
        )
        self.min_sample_split = max(min_sample_split, sample_split_lower_bound)
        self.fixed_lag = fixed_lag
        self.external_lag = external_lag
        self.stopping_criteria = stopping_criteria
        self.error_threshold = error_threshold
        self.seq_significance = seq_significance
        self.significance_divider = significance_divider
        self.verbose = verbose

        # Initialize any other common variables needed for SETAR models

    def fit(self, X, y):
        raise NotImplementedError("Abstract class")

    def predict(self, X):
        raise NotImplementedError("Abstract class")


class SETARTree(BaseSETAR):
    def __init__(
        self,
        lag: int,
        forecast_horizon: int,
        exogenous_covariates: List[str] = None,
        depth: int = 8,
        significance: float = 0.05,
        min_sample_split: int = 5,
        fixed_lag: bool = False,
        external_lag: int = 1,
        stopping_criteria: str = "both",
        error_threshold: float = 0.05,
        seq_significance: bool = True,  # whether to apply significance_divider
        significance_divider: float = 2,
        verbose: int = 0,
    ):

        super().__init__(
            lag=lag,
            forecast_horizon=forecast_horizon,
            exogenous_covariates=exogenous_covariates,
            depth=depth,
            significance=significance,
            min_sample_split=min_sample_split,
            fixed_lag=fixed_lag,
            external_lag=external_lag,
            stopping_criteria=stopping_criteria,
            error_threshold=error_threshold,
            seq_significance=seq_significance,
            significance_divider=significance_divider,
            verbose=verbose,
        )

        # Initialize any SETARTree specific variable
        self.start_con = {
            "nTh": 15
        }  # Number of thresholds when making each split to define optimal lag
        self._tree = []  # Stores the nodes in tree in each level
        self._th_lags = []  # Stores the optimal lags used during splitting
        self._thresholds = (
            []
        )  # Stores the optimal thresholds used during splitting
        self._level_errors = []
        self._node_data = (
            []
        )  # Root node contains the training instances coming from all series
        self._split_info = []
        self._exogenous_var_ids = None
        self._column_length = None
        self._leaf_trained_models = None
        self._num_of_leaf_instances = None
        self._final_trained_model = None
        self._fit = False

    def fit(self, X, y):
        """
        TODO:
        - Do something with the RSS criterion.
        - Remove as much as possible the dictionnaries outputs.
        - Investigate the presence of ths = [1] in recheck.
        - Maybe split X and y
        - Have the model to support exogenous variables
        - Separate model into multiple sub functions
        """

        # checks
        data = pd.concat([y, X], axis=1)
        # TODO: Data checks
        exo_var_length = (
            0
            if self.exogenous_covariates is None
            else len(self.exogenous_covariates)
        )
        expected_col_length = 1 + self.lag + exo_var_length  # y + Lags + exo
        column_length = data.shape[1]
        assert (
            column_length == expected_col_length
        ), f"expected {expected_col_length} columns but got {column_length}"

        if self.verbose > 1:
            print(data)
        # Start timestamp
        all_start_time = time.time()

        # Set list of defaults (reset to avoid issues)

        self._column_length = column_length
        self._tree = []
        self._th_lags = []
        self._thresholds = []
        self._level_errors = []
        self._node_data = [data]
        self._split_info = [1]
        self._exogenous_var_ids = None

        # Identify the column indexes of exogenous covariates
        if self.exogenous_covariates is not None:
            self._exogenous_var_ids = range(self._lag + 1, self._column_length)

        for d in range(1, self.depth + 1):
            if self.verbose > 0:
                print(f"Depth: {d}")

            level_th_lags = []
            level_thresholds = []
            level_nodes = []
            level_significant_node_count = 0
            level_split_info = []

            for n, node in enumerate(self._node_data):
                if self.verbose > 0:
                    print("n=", n)

                # print("Node (begin)")
                # print(node)

                best_cost = float("inf")
                th = None
                th_lag = None

                if (len(node) > self.min_sample_split) and self._split_info[
                    n
                ] == 1:

                    if self.fixed_lag:
                        lg = self.external_lag if self.external_lag > 0 else 1
                        # print("X")
                        # display(node.iloc[:, 1:])
                        # print("y")
                        # display(node.iloc[:, 0])
                        # print("ix")
                        # display(node.iloc[:, lg + 1])
                        # print(f"k={start_con['nTh']}, lag={lag}")

                        ss_output = find_cut_point(
                            node.iloc[:, 1:].to_numpy(),
                            node.iloc[:, 0].to_numpy(),
                            node.iloc[:, lg].to_numpy(),
                            self.start_con["nTh"],
                            self.lag,
                            criterion="RSS",
                        )
                        cost = ss_output["RSS_left"] + ss_output["RSS_right"]

                        if cost <= best_cost:
                            best_cost = cost
                            th = ss_output["cut_point"]
                            th_lag = lg
                            # th_lag = X.columns[lg]
                    else:
                        for lg in range(
                            1, self._column_length
                        ):  # we access corresponding col at index `lg`
                            # -> no ambiguity
                            if self.verbose > 0:
                                print(f"Lag {lg}")
                            # print("X")
                            # display(node.iloc[:, 1:])
                            # print("y")
                            # display(node.iloc[:, 0])
                            # print("ix")
                            # display(node.iloc[:, lg])
                            # print(f"k={start_con['nTh']}, lag={lag}")

                            # Finding the optimal lag and threshold
                            # that should be used for splitting
                            # Optimized grid search
                            ss_output = find_cut_point(
                                node.iloc[:, 1:].to_numpy(),
                                node.iloc[:, 0].to_numpy(),
                                node.iloc[:, lg].to_numpy(),
                                self.start_con["nTh"],
                                self.lag,
                                criterion="RSS",
                            )
                            cost = (
                                ss_output["RSS_left"] + ss_output["RSS_right"]
                            )
                            recheck = ss_output["need_recheck"]

                            if recheck > round(self.start_con["nTh"] * 0.6):
                                # If optimized grid search fails, then try
                                # with exhaustive search (SS)
                                if self.exogenous_var_ids is not None and (
                                    lg in self.exogenous_var_ids
                                ):
                                    ths = [1]  # strange, why?
                                else:
                                    ths = np.linspace(
                                        node.iloc[:, lg].min(),
                                        node.iloc[:, lg].max(),
                                        num=self.start_con["nTh"],
                                    )

                                for th_val in ths:
                                    cost = SS(th_val, node, lg)

                                    if cost <= best_cost:
                                        best_cost = cost
                                        th = th_val
                                        th_lag = lg
                                        # th_lag = X.columns[lg]
                            else:
                                if cost <= best_cost:
                                    best_cost = cost
                                    th = ss_output["cut_point"]
                                    th_lag = lg
                                    # th_lag = X.columns[lg]

                    if best_cost != float("inf"):
                        split_nodes = create_split(node, th_lag, th)
                        # print("Split nodes")
                        # print(split_nodes)

                        # Check whether making the split is worth enough
                        if self.stopping_criteria == "lin_test":
                            is_significant = check_linearity(
                                node,
                                split_nodes,
                                self._column_length - 1,
                                self.significance,
                            )
                        elif self.stopping_criteria == "error_imp":
                            is_significant = check_error_improvement(
                                node, split_nodes, self.error_threshold
                            )
                        elif self.stopping_criteria == "both":
                            is_significant = check_linearity(
                                node,
                                split_nodes,
                                self._column_length - 1,
                                self.significance,
                            ) and check_error_improvement(
                                node, split_nodes, self.error_threshold
                            )

                        if is_significant:
                            level_th_lags.append(th_lag)
                            level_thresholds.append(th)
                            level_split_info.extend([1, 1])
                            level_significant_node_count += 1

                            level_nodes.extend(
                                list(split_nodes.values())
                            )  # TODO: likely to be very inefficient,
                            # need change ASAP
                        else:
                            level_th_lags.append(0)
                            level_thresholds.append(0)
                            level_split_info.append(0)
                            level_nodes.append(node)
                    else:
                        level_th_lags.append(0)
                        level_thresholds.append(0)
                        level_split_info.append(0)

                        level_nodes.append(node)
                else:
                    level_th_lags.append(0)
                    level_thresholds.append(0)
                    level_split_info.append(0)

                    level_nodes.append(node)

            #     print("Node (end)")
            #     print(node)
            # print("Level nodes")
            # print(level_nodes)

            if level_significant_node_count > 0:
                self._tree.append(level_nodes)
                self._thresholds.append(level_thresholds)
                self._th_lags.append(level_th_lags)
                self._node_data = self._tree[-1]
                self._split_info = level_split_info

                if (
                    self.seq_significance
                ):  # Define the significance for the next level of the tree
                    self.significance /= self.significance_divider
            else:
                # If all nodes in a particular tree level
                # are not further split, then stop
                break

        # Making sure that the bottom layer of the three contains
        # only leaf nodes
        if len(self._tree) > 0:
            thresholds = self._thresholds[-1]
            th_lags = self._th_lags[-1]

            threhsolds_nonzero = len(np.nonzero(thresholds))
            th_lags_nonzero = len(np.nonzero(th_lags))
            assert (
                threhsolds_nonzero == th_lags_nonzero
            ), f"""inconsistency error, last threhsolds layer contains
                {threhsolds_nonzero} non zero values but last th_lags layer
                contains {th_lags_nonzero} non zero values"""
            if threhsolds_nonzero > 0:
                self._thresholds.append(
                    [0] * (len(thresholds) + threhsolds_nonzero)
                )
                self._th_lags.append([0] * (len(th_lags) + th_lags_nonzero))

        # Model training
        # Check whether the tree has any nodes. If not, train a single
        # pooled regression model.
        if len(self._tree) > 0:
            leaf_nodes = self._tree[-1]
            self._leaf_trained_models = []
            self._num_of_leaf_instances = []

            # Train a linear model for each leaf node
            for leaf_node in leaf_nodes:
                model = fit_global_model(leaf_node)["model"]
                self._leaf_trained_models.append(model)
                self._num_of_leaf_instances.append(len(leaf_node))
        else:
            self._final_trained_model = fit_global_model(data)["model"]

        self._fit = True
        print(f"Fit model in {time.time()-all_start_time:.3f} seconds")

    def predict(self, X, futr_var=None):
        """We suppose that X is tabular with the lag variables and the
        exogenous variables that we know also in the future (otherwise we
        cannot perform autoregressive forecasting)
        """
        assert isinstance(X, pd.Series) or isinstance(X, pd.DataFrame)
        if isinstance(X, pd.Series):
            X = X.to_frame().T.copy()
        else:
            X = X.copy()

        n_xrows, n_xcols = X.shape
        assert self._fit, "model is not trained"
        assert n_xrows == 1, "batch generation is not supported yet"
        assert (
            n_xcols == self.lag
        ), f"""number of columns must match the training data,
            expected {self.lag} but got {n_xcols}"""

        if self.exogenous_covariates:
            assert isinstance(futr_var, pd.DataFrame)
            n_futr_rows, n_futr_cols = futr_var.shape
            assert (
                n_futr_rows == self.forecast_horizon
            ), f"""number of futur rows must match `forecasting_horizon`,
                expected {self.forecast_horizon} but got {n_futr_rows}"""
            assert n_futr_cols == len(
                self.exogenous_covariates
            ), f"""number of futur rows must match the training data,
                expected {len(self.exogenous_covariates)} but got
                {n_futr_cols}"""

        # Forecasting start timestamp
        forecasting_start_time = time.time()

        rolling_data = X
        # X must contain ['lag1','lag2',...]

        forecasts = []

        for horizon in range(self.forecast_horizon):
            if self.verbose > 0:
                print(f"Horizon = {horizon}")

            if self._tree:

                leaf_index = get_leaf_index(
                    rolling_data, self._th_lags, self._thresholds
                )

                # print("leaf index")
                # print(leaf_index)
                # get prediction from leaf model
                leaf_model = self._leaf_trained_models[leaf_index]
                predictions = leaf_model.predict(rolling_data)

            else:
                # use global model
                predictions = self._final_trained_model.predict(rolling_data)

            forecasts.append(predictions)

            # update data
            if horizon < self.forecast_horizon:

                rolling_data_array = np.column_stack(
                    [predictions, rolling_data.iloc[:, :-1]]
                )
                rolling_data.loc[
                    :
                ] = rolling_data_array  # may need tweaking with DataFrames

        forecasts = np.column_stack(forecasts)

        print(
            f"""Predictions made in {time.time()-forecasting_start_time:.3f}
              seconds"""
        )
        return forecasts
