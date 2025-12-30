import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split


def split_dataset_pd(
    df,
    X,
    y=None,  # Made optional with default None
    train_val_test_ratio=(0.7, 0.3, 0),
    X_duplicated=False,
    train_val_test_name=("train", "val", "test"),
    dataset_col_name="datasets",
    random_state=1,
):
    """
    df: 数据集
    X: 样本列名
    y: 标签列名 (可选, 如果提供则进行分层抽样, 否则随机抽样)
    train_val_test_ratio: 训练集、验证集、测试集的比例, 测试集可为0
    X_duplicated: 是否考虑X列的重复值
    train_val_test_name: 训练集、验证集、测试集的名称
    dataset_col_name: 数据集列名
    random_state: 随机种子
    """
    train_size, val_size, test_size = train_val_test_ratio

    train_val_ratio = (val_size + test_size) / (train_size + val_size + test_size)
    val_test_ratio = (
        (test_size) / (val_size + test_size) if (val_size + test_size) > 0 else 0
    )

    # Prepare data for splitting
    if X_duplicated:
        if y is not None:
            df_stratify = (
                df.groupby(X)[y].agg(lambda x: pd.Series.mode(x)[0]).reset_index()
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

    # First split: train vs (val + test)
    stratify_col = df_stratify["y"] if y is not None else None
    train_x, tmp_x = train_test_split(
        df_stratify.x,
        test_size=train_val_ratio,
        stratify=stratify_col,
        random_state=random_state,
    )

    # Second split: val vs test
    if val_test_ratio == 0:
        val_x = tmp_x
        test_x = []
    else:
        if y is not None:
            tmp_y = df_stratify.loc[df_stratify.x.isin(tmp_x)]["y"]
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

    # Assign dataset labels
    train_mask = df[X].isin(train_x)
    val_mask = df[X].isin(val_x)
    test_mask = df[X].isin(test_x)

    df[dataset_col_name] = pd.NA

    train, val, test = train_val_test_name
    df.loc[train_mask, dataset_col_name] = train
    df.loc[val_mask, dataset_col_name] = val
    df.loc[test_mask, dataset_col_name] = test

    return df


def split_dataset_pl(
    df: pl.DataFrame,
    X: str,
    y: str = None, 
    train_val_test_ratio=(0.7, 0.3, 0),
    X_duplicated=False,
    train_val_test_name=("train", "val", "test"),
    dataset_col_name="dataset",
    random_state=1,
):
    """
    df: 数据集
    X: 样本列名
    y: 标签列名 (可选，若提供则进行分层抽样，否则随机抽样)
    train_val_test_ratio: 训练集、验证集、测试集的比例, 测试集可为0
    X_duplicated: 是否考虑X列的重复值
    train_val_test_name: 训练集、验证集、测试集的名称
    dataset_col_name: 数据集列名
    random_state: 随机种子
    """
    train_size, val_size, test_size = train_val_test_ratio

    train_val_ratio = (val_size + test_size) / (train_size + val_size + test_size)
    val_test_ratio = (
        test_size / (val_size + test_size) if (val_size + test_size) > 0 else 0
    )

    # Prepare data for splitting
    if X_duplicated:
        if y is not None:
            # Group by X and get mode of y
            df_stratify = (
                df.group_by(X)
                .agg(pl.col(y).mode().list.first().alias("y"))
                .rename({X: "x"})
            )
        else:
            # Just get unique X values
            df_stratify = df.select(pl.col(X).unique().alias("x"))
    else:
        if y is not None:
            df_stratify = df.select([pl.col(X).alias("x"), pl.col(y).alias("y")])
        else:
            df_stratify = df.select(pl.col(X).alias("x"))

    # Convert to lists for sklearn
    stratify_x = df_stratify.get_column("x").to_list()
    stratify_y = df_stratify.get_column("y").to_list() if y is not None else None

    # First split: train vs (val + test)
    if y is not None:
        train_x, tmp_x, _, tmp_y = train_test_split(
            stratify_x,
            stratify_y,
            test_size=train_val_ratio,
            stratify=stratify_y,
            random_state=random_state,
        )
    else:
        train_x, tmp_x = train_test_split(
            stratify_x,
            test_size=train_val_ratio,
            random_state=random_state,
        )
        tmp_y = None

    # Second split: val vs test
    if val_test_ratio == 0:
        val_x = tmp_x
        test_x = []
    else:
        if y is not None:
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

    train_name, val_name, test_name = train_val_test_name

    # Create the dataset column using when/then/otherwise
    df = df.with_columns(
        pl.when(pl.col(X).is_in(train_x))
        .then(pl.lit(train_name))
        .when(pl.col(X).is_in(val_x))
        .then(pl.lit(val_name))
        .when(pl.col(X).is_in(test_x))
        .then(pl.lit(test_name))
        .otherwise(pl.lit(None))
        .alias(dataset_col_name)
    )

    return df
