from typing import Any, Dict, List, Optional, Tuple

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

# Type aliases
Ratio = Tuple[float, float, float]
Names = Tuple[str, str, str]


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
        test_size / (val_size + test_size) if (val_size + test_size) > 0 else 0.0
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


def _perform_fold_splits(
    x_values: List,
    y_values: Optional[List],
    num_folds: int,
    random_state: int,
) -> Dict[Any, int]:
    """
    Assign fold indices to x_values using KFold or StratifiedKFold.

    Parameters
    ----------
    x_values : list
        List of unique sample identifiers.
    y_values : list, optional
        List of labels for stratification. If provided, uses StratifiedKFold.
    num_folds : int
        Number of folds.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Mapping from x_value to fold index (0 to num_folds-1).
    """
    if y_values is not None:
        kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
        splits = kfold.split(x_values, y_values)
    else:
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
        splits = kfold.split(x_values)

    fold_map = {}
    for fold_idx, (_, val_indices) in enumerate(splits):
        for idx in val_indices:
            fold_map[x_values[idx]] = fold_idx

    return fold_map
