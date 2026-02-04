from typing import List, Optional, Tuple

from sklearn.model_selection import train_test_split

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
