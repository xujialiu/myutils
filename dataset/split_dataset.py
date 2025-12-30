from typing import List, Optional, Tuple, Union

import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split

# Type aliases
Ratio = Tuple[float, float, float]
Names = Tuple[str, str, str]


# =============================================================================
# Utility Functions
# =============================================================================

def _validate_ratio(ratio: Ratio) -> None:
    """Validate train/val/test ratio."""
    if len(ratio) != 3:
        raise ValueError("Ratio must have exactly 3 values (train, val, test)")
    if any(r < 0 for r in ratio):
        raise ValueError("All ratios must be non-negative")
    if sum(ratio) <= 0:
        raise ValueError("Sum of ratios must be positive")


def _calculate_split_ratios(ratio: Ratio) -> Tuple[float, float]:
    """
    Calculate split ratios for two-stage splitting.
    
    Returns
    -------
    tuple
        (train_val_ratio, val_test_ratio)
    """
    train_size, val_size, test_size = ratio
    total = train_size + val_size + test_size

    train_val_ratio = (val_size + test_size) / total
    val_test_ratio = (
        test_size / (val_size + test_size) 
        if (val_size + test_size) > 0 
        else 0.0
    )
    return train_val_ratio, val_test_ratio


def _perform_splits(
    x_values: List,
    y_values: Optional[List],
    train_val_ratio: float,
    val_test_ratio: float,
    random_state: int,
) -> Tuple[List, List, List]:
    """
    Perform train/val/test splits using sklearn.

    Returns
    -------
    tuple
        (train_x, val_x, test_x) lists
    """
    # Edge case: all samples go to train (no split needed)
    if train_val_ratio == 0:
        return x_values, [], []

    # First split: train vs (val + test)
    if y_values is not None:
        train_x, tmp_x, _, tmp_y = train_test_split(
            x_values,
            y_values,
            test_size=train_val_ratio,
            stratify=y_values,
            random_state=random_state,
        )
    else:
        train_x, tmp_x = train_test_split(
            x_values,
            test_size=train_val_ratio,
            random_state=random_state,
        )
        tmp_y = None

    # Second split: val vs test
    if val_test_ratio == 0:
        val_x = tmp_x
        test_x = []
    elif tmp_y is not None:
        val_x, test_x = train_test_split(
            tmp_x,
            test_size=val_test_ratio,
            stratify=tmp_y,
            random_state=random_state + 1,
        )
    else:
        val_x, test_x = train_test_split(
            tmp_x,
            test_size=val_test_ratio,
            random_state=random_state + 1,
        )

    return train_x, val_x, test_x


# =============================================================================
# Pandas Implementation
# =============================================================================

def _prepare_stratify_df_pd(
    df: pd.DataFrame,
    X: str,
    y: Optional[str],
    X_duplicated: bool,
) -> pd.DataFrame:
    """Prepare stratification dataframe for pandas."""
    if X_duplicated:
        if y is not None:
            df_stratify = (
                df.groupby(X)[y]
                .agg(lambda x: pd.Series.mode(x).iloc[0])
                .reset_index()
            )
            df_stratify.columns = ["x", "y"]
        else:
            df_stratify = df[[X]].drop_duplicates().reset_index(drop=True)
            df_stratify.columns = ["x"]
    else:
        if y is not None:
            df_stratify = df[[X, y]].copy()
            df_stratify.columns = ["x", "y"]
        else:
            df_stratify = df[[X]].copy()
            df_stratify.columns = ["x"]
    
    return df_stratify


def split_dataset_pd(
    df: pd.DataFrame,
    X: str,
    y: Optional[str] = None,
    train_val_test_ratio: Ratio = (0.7, 0.3, 0.0),
    X_duplicated: bool = False,
    train_val_test_name: Names = ("train", "val", "test"),
    dataset_col_name: str = "dataset",
    random_state: int = 1,
) -> pd.DataFrame:
    """
    Split a pandas DataFrame into train/validation/test sets.

    Parameters
    ----------
    df : pd.DataFrame
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
    pd.DataFrame
        DataFrame with added dataset split column.
    """
    _validate_ratio(train_val_test_ratio)
    train_val_ratio, val_test_ratio = _calculate_split_ratios(train_val_test_ratio)

    # Prepare stratification data
    df_stratify = _prepare_stratify_df_pd(df, X, y, X_duplicated)
    
    x_values = df_stratify["x"].tolist()
    y_values = df_stratify["y"].tolist() if y is not None else None

    # Perform splits
    train_x, val_x, test_x = _perform_splits(
        x_values, y_values, train_val_ratio, val_test_ratio, random_state
    )

    # Assign dataset labels
    train_name, val_name, test_name = train_val_test_name
    
    df = df.copy()
    df[dataset_col_name] = pd.NA
    df.loc[df[X].isin(train_x), dataset_col_name] = train_name
    df.loc[df[X].isin(val_x), dataset_col_name] = val_name
    df.loc[df[X].isin(test_x), dataset_col_name] = test_name

    return df


# =============================================================================
# Polars Implementation
# =============================================================================

def _prepare_stratify_df_pl(
    df: pl.DataFrame,
    X: str,
    y: Optional[str],
    X_duplicated: bool,
) -> pl.DataFrame:
    """Prepare stratification dataframe for polars."""
    if X_duplicated:
        if y is not None:
            df_stratify = (
                df.group_by(X)
                .agg(pl.col(y).mode().first().alias("y"))
                .rename({X: "x"})
            )
        else:
            df_stratify = df.select(pl.col(X).unique().alias("x"))
    else:
        if y is not None:
            df_stratify = df.select([
                pl.col(X).alias("x"), 
                pl.col(y).alias("y")
            ])
        else:
            df_stratify = df.select(pl.col(X).alias("x"))
    
    return df_stratify


def split_dataset_pl(
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
    df_stratify = _prepare_stratify_df_pl(df, X, y, X_duplicated)
    
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


# =============================================================================
# Unified Interface
# =============================================================================

def split_dataset(
    df: Union[pd.DataFrame, pl.DataFrame],
    X: str,
    y: Optional[str] = None,
    train_val_test_ratio: Ratio = (0.7, 0.3, 0.0),
    X_duplicated: bool = False,
    train_val_test_name: Names = ("train", "val", "test"),
    dataset_col_name: str = "dataset",
    random_state: int = 1,
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Split a DataFrame into train/validation/test sets.

    Automatically detects pandas or polars DataFrame and calls
    the appropriate implementation.

    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
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
    pd.DataFrame or pl.DataFrame
        DataFrame with added dataset split column.

    Examples
    --------
    >>> df = pd.DataFrame({"id": range(100), "label": [0]*50 + [1]*50})
    >>> result = split_dataset(df, X="id", y="label")
    >>> result["dataset"].value_counts()
    """
    kwargs = {
        "X": X,
        "y": y,
        "train_val_test_ratio": train_val_test_ratio,
        "X_duplicated": X_duplicated,
        "train_val_test_name": train_val_test_name,
        "dataset_col_name": dataset_col_name,
        "random_state": random_state,
    }

    if isinstance(df, pd.DataFrame):
        return split_dataset_pd(df, **kwargs)
    elif isinstance(df, pl.DataFrame):
        return split_dataset_pl(df, **kwargs)
    else:
        raise TypeError(f"Unsupported DataFrame type: {type(df).__name__}")