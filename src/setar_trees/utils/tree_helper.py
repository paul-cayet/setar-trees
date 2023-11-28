import numpy as np
import pandas as pd
from scipy.stats import f
from sklearn.linear_model import LinearRegression


# A function to split a parent node into child nodes
def create_split(data, conditional_lag, threshold):
    """
    TODO: Change it to be Numpy comptatible
    """
    left_node = data[data.iloc[:, conditional_lag] < threshold]
    right_node = data[data.iloc[:, conditional_lag] >= threshold]
    return {"left_node": left_node, "right_node": right_node}


# A function to traverse through the tree based on the
# used thresholds and lags during splitting
# def tree_traverse(instance, split, threshold):
#     """
#     TODO: Change it to be Numpy comptatible
#     """
#     direction = "left"
#     if instance[split] >= threshold:
#         direction = "right"
#     return direction


def tree_traverse(instance, split, threshold):
    """
    TODO: Change it to be Numpy comptatible
    """
    direction = "left"
    if (
        instance.iloc[0, split - 1] >= threshold
    ):  # -1 because we don't have the y variable anymore
        direction = "right"
    return direction


def get_leaf_index(instance, splits, thresholds, verbose=0):
    """
    TODO:
    - Rename variables
    - Make the code numpy friendly
    """
    current_split = 0  # Start with the root node index as 0
    divide_factor = 2

    for sp in range(len(splits)):
        # Check if the current split contains zeros indicating leaf nodes
        if 0 in splits[sp]:
            zeros = [i for i, x in enumerate(splits[sp]) if x == 0]
            change_count = sum(1 for zero in zeros if zero < current_split)
            next_possible_splits = list(
                range(
                    current_split * divide_factor,
                    (current_split + 1) * divide_factor,
                )
            )
            next_possible_splits = [
                x - change_count for x in next_possible_splits
            ]
            if splits[sp][current_split] == 0:
                current_split = next_possible_splits[0]
                # print('zero+split=0',next_possible_splits, current_split)
            else:
                direction = tree_traverse(
                    instance,
                    splits[sp][current_split],
                    thresholds[sp][current_split],
                )
                current_split = (
                    next_possible_splits[0]
                    if direction == "left"
                    else next_possible_splits[1]
                )
                # print('zero+split!=0',next_possible_splits, current_split)
        else:
            direction = tree_traverse(
                instance,
                splits[sp][current_split],
                thresholds[sp][current_split],
            )
            next_possible_splits = list(
                range(
                    current_split * divide_factor,
                    (current_split + 1) * divide_factor,
                )
            )

            current_split = (
                next_possible_splits[0]
                if direction == "left"
                else next_possible_splits[1]
            )
            # print('no zero',next_possible_splits, current_split)
    return current_split


def fit_global_model(data):
    """
    TODO:
    - check if this what we really need
    """
    # Implement the logic for fitting a model to the data
    # For demo purposes, we'll just return zeros
    X, y = data.iloc[:, 1:], data.iloc[:, 0]
    model = LinearRegression().fit(X, y)
    preds = model.predict(X)
    return {"preds": preds, "model": model}


# A function to calculate Sum of Squared Errors (SSE)
def SS(p, train_data, current_lg):
    """
    TODO: Refactor lol, p is threshold, current_lg -> current lag
    + do something about the `fit_global_model` because it is quite
    horrible.
    """
    splitted_nodes = create_split(train_data, current_lg, p)
    left = splitted_nodes["left_node"]
    right = splitted_nodes["right_node"]

    cost = float("inf")
    if not left.empty and not right.empty:
        # Placeholder for the fit_global_model function
        residuals_l = left["y"] - fit_global_model(left)["preds"]
        residuals_r = right["y"] - fit_global_model(right)["preds"]
        current_residuals = np.concatenate([residuals_l, residuals_r])
        cost = np.sum(current_residuals**2)

    return cost


# A function to check whether there exists a remaining
# non-linearity in the parent node instances
def check_linearity(parent_node, child_nodes, lag, significance, verbose=0):

    is_significant = True

    ss0 = np.sum(
        (parent_node["y"] - fit_global_model(parent_node)["preds"]) ** 2
    )
    if verbose > 0:
        print(f"SSO={ss0:.3f}")
    if ss0 == 0:
        is_significant = False
    else:
        train_residuals = np.array([])
        for child_node in child_nodes.values():
            train_residuals = np.concatenate(
                [
                    train_residuals,
                    (child_node["y"] - fit_global_model(child_node)["preds"]),
                ]
            )

        ss1 = np.sum(train_residuals**2)
        # ^if ss1>ss0 (worse than parent), then p_val = 1, split is useless

        # Compute F-statistic (matches the R version)
        f_stat = ((ss0 - ss1) / (lag + 1)) / (
            ss1 / (len(parent_node) - 2 * lag - 2)
        )
        p_value = f.sf(f_stat, lag + 1, len(parent_node) - 2 * lag - 2)

        if p_value > significance:
            is_significant = False

    return is_significant


# A function to check whether a considerable error reduction
# can be gained by splitting a parent node into child nodes
def check_error_improvement(parent_node, child_nodes, error_threshold):
    is_improved = True

    ss0 = np.sum(
        (parent_node["y"] - fit_global_model(parent_node)["preds"]) ** 2
    )

    if ss0 == 0:
        is_improved = False
    else:
        train_residuals = np.array([])
        for child_node in child_nodes.values():
            train_residuals = np.concatenate(
                [
                    train_residuals,
                    (child_node["y"] - fit_global_model(child_node)["preds"]),
                ]
            )

        ss1 = np.sum(train_residuals**2)

        improvement = (ss0 - ss1) / ss0

        if improvement < error_threshold:
            is_improved = False

    return is_improved
