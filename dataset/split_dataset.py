from typing import Optional

import polars as pl

from ._split_utils import (
    Names,
    Ratio,
    _calculate_split_ratios,
    _perform_fold_splits,
    _perform_splits,
    _validate_ratio,
)


def _prepare_stratify_df(
    df: pl.DataFrame,
    X: str,
    y: Optional[str],
    X_duplicated: bool,
) -> pl.DataFrame:
    """Prepare stratification dataframe for polars."""
    if X_duplicated:
        if y is not None:
            df_stratify = (
                df.group_by(X).agg(pl.col(y).mode().first().alias("y")).rename({X: "x"})
            )
        else:
            df_stratify = df.select(pl.col(X).unique().alias("x"))
    else:
        if y is not None:
            df_stratify = df.select([pl.col(X).alias("x"), pl.col(y).alias("y")])
        else:
            df_stratify = df.select(pl.col(X).alias("x"))

    return df_stratify


def split_dataset(
    df: pl.DataFrame,
    X: str,
    y: Optional[str] = None,
    train_val_test_ratio: Ratio = (0.7, 0.3, 0.0),
    X_duplicated: bool = False,
    train_val_test_name: Names = ("train", "val", "test"),
    dataset_col_name: str = "dataset",
    random_state: int = 1,
) -> pl.DataFrame:
    """
    Split a Polars DataFrame into train/validation/test sets.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataset.
    X : str
        Column name for sample identifier.
    y : str, optional
        Column name for labels. If provided, stratified sampling is used.
    train_val_test_ratio : tuple of float, default=(0.7, 0.3, 0.0)
        Ratios for (train, validation, test) sets. Test can be 0.
    X_duplicated : bool, default=False
        Whether to handle duplicated values in X column by grouping.
    train_val_test_name : tuple of str, default=("train", "val", "test")
        Names for train, validation, and test sets.
    dataset_col_name : str, default="dataset"
        Name of the output column indicating dataset split.
    random_state : int, default=1
        Random seed for reproducibility.

    Returns
    -------
    pl.DataFrame
        DataFrame with added dataset split column.
    """
    _validate_ratio(train_val_test_ratio)
    train_val_ratio, val_test_ratio = _calculate_split_ratios(train_val_test_ratio)

    # Prepare stratification data
    df_stratify = _prepare_stratify_df(df, X, y, X_duplicated)

    x_values = df_stratify.get_column("x").to_list()
    y_values = df_stratify.get_column("y").to_list() if y is not None else None

    # Perform splits
    train_x, val_x, test_x = _perform_splits(
        x_values, y_values, train_val_ratio, val_test_ratio, random_state
    )

    # Assign dataset labels
    train_name, val_name, test_name = train_val_test_name

    return df.with_columns(
        pl.when(pl.col(X).is_in(train_x))
        .then(pl.lit(train_name))
        .when(pl.col(X).is_in(val_x))
        .then(pl.lit(val_name))
        .when(pl.col(X).is_in(test_x))
        .then(pl.lit(test_name))
        .otherwise(pl.lit(None))
        .alias(dataset_col_name)
    )


def split_dataset_folds(
    df: pl.DataFrame,
    X: str,
    y: Optional[str] = None,
    X_duplicated: bool = False,
    num_folds: int = 5,
    folds_col_name: str = "fold",
    random_state: int = 1,
) -> pl.DataFrame:
    """
    Assign each sample to one of num_folds folds for k-fold cross-validation.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataset.
    X : str
        Column name for sample identifier.
    y : str, optional
        Column name for labels. If provided, uses StratifiedKFold.
    X_duplicated : bool, default=False
        Whether to handle duplicated values in X column by grouping.
    num_folds : int, default=5
        Number of folds for cross-validation.
    folds_col_name : str, default="fold"
        Name of the output column indicating fold assignment.
    random_state : int, default=1
        Random seed for reproducibility.

    Returns
    -------
    pl.DataFrame
        DataFrame with added fold column (integers 0 to num_folds-1).
    """
    df_stratify = _prepare_stratify_df(df, X, y, X_duplicated)

    x_values = df_stratify.get_column("x").to_list()
    y_values = df_stratify.get_column("y").to_list() if y is not None else None

    fold_map = _perform_fold_splits(x_values, y_values, num_folds, random_state)

    return df.with_columns(pl.col(X).replace(fold_map).alias(folds_col_name))
