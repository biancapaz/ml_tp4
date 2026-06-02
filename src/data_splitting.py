import numpy as np

def random_stratified_split(df, target_col="label", test_ratio=0.2, random_state=42):
    """Splits a DataFrame into training and validation sets randomly mantaining classes' proportion.

    Args:
        df (pd.DataFrame): The input DataFrame to split.
        target_col (str, optional): Column name of the target class label. Defaults to "label".
        test_ratio (float, optional): The proportion of data to use for testing. Defaults to 0.2.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (train_set, val_set) A tuple containing the training DataFrame and validation DataFrame.
    """
    rng = np.random.default_rng(random_state) # set a seed for reproducibility
    train_idx, val_idx = [], []

    for cls in df[target_col].unique():
        cls_idx = df.index[df[target_col] == cls].to_numpy().copy() # class idxs
        rng.shuffle(cls_idx) # shuffle idxs
        cut = int(len(cls_idx) * (1 - test_ratio)) # up to what idx
        train_idx.extend(cls_idx[:cut])
        val_idx.extend(cls_idx[cut:])

    train_set = df.loc[train_idx].reset_index(drop=True)
    val_set   = df.loc[val_idx].reset_index(drop=True)
    return train_set, val_set