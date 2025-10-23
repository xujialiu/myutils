import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(
    df,
    X,
    y,
    train_val_test_ratio=(0.7, 0.3, 0),
    X_duplicated=False,
    train_val_test_name=("train", "val", "test"),
    dataset_col_name="datasets",
    random_state=1,
):
    """
    df: 数据集
    X: 样本列名
    y: 标签列名
    train_val_test_ratio: 训练集、验证集、测试集的比例, 测试集可为0
    X_duplicated: 是否考虑X列的重复值
    train_val_test_name: 训练集、验证集、测试集的名称
    dataset_col_name: 数据集列名
    random_state: 随机种子
    """
    train_size, val_size, test_size = train_val_test_ratio

    train_val_ratio = (val_size + test_size) / (train_size + val_size + test_size)
    val_test_ratio = (test_size) / (val_size + test_size)

    if X_duplicated:
        df_stratify = df.groupby(X)[y].agg(lambda x: pd.Series.mode(x)[0]).reset_index()
    else:
        df_stratify = df[[X, y]]

    df_stratify.columns = ["x", "y"]

    train_x, tmp_x = train_test_split(
        df_stratify.x,
        test_size=train_val_ratio,
        stratify=df_stratify.y,
        random_state=random_state,  # 确保可重复性
    )

    tmp_y = df_stratify.loc[df_stratify.x.isin(tmp_x)]["y"]

    if val_test_ratio == 0:
        val_x = tmp_x
        test_x = []
    else:
        val_x, test_x = train_test_split(
            tmp_x,
            test_size=val_test_ratio,
            stratify=tmp_y,
            random_state=random_state + 1,  # 确保可重复性
        )

    train_mask = df[X].isin(train_x)
    val_mask = df[X].isin(val_x)
    test_mask = df[X].isin(test_x)

    df[dataset_col_name] = pd.NA

    train, val, test = train_val_test_name
    df.loc[train_mask, dataset_col_name] = train
    df.loc[val_mask, dataset_col_name] = val
    df.loc[test_mask, dataset_col_name] = test
    return df
