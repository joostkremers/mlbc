import numbers

import pandas as pd
import numpy as np


def read_data(file):
    df = pd.read_csv(file)
    return df


def clean_alphanum_data(df):
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    string_columns = list(df.dtypes[df.dtypes == 'object'].index)
    for col in string_columns:
        df[col] = df[col].str.lower().str.replace(' ', '_')


def split_data_frame(df, split=0.2, seed=None):
    n = len(df)

    n_val = int(split * n)
    n_test = int(split * n)
    n_train = n - (n_val + n_test)

    idx = np.arange(n)
    if isinstance(seed, numbers.Number):
        np.random.seed(seed)
    np.random.shuffle(idx)

    df_shuffled = df.iloc[idx]

    df_train = df_shuffled.iloc[:n_train].copy()
    df_val = df_shuffled.iloc[n_train:n_train+n_val].copy()
    df_test = df_shuffled.iloc[n_train+n_val:].copy()

    return df_train, df_val, df_test


def prepare_X(df, base):
    df = df.copy()
    features = base.copy()

    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values

    return X


def linear_regression(X, y, r=0.0):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    reg = r * np.eye(XTX.shape[0])
    XTX = XTX + reg

    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)

    return w[0], w[1:]


def rmse(y, y_pred):
    error = y_pred - y
    mse = (error ** 2).mean()
    return np.sqrt(mse)


# base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']

# df = read_data('data.csv')

# clean_alphanum_data(df)

# df_train, df_val, df_test = split_data_frame(df, split=0.2, seed=2)

# # Apply log transformation to the output values and store these separately.
# y_train = np.log1p(df_train.msrp.values)
# y_val = np.log1p(df_val.msrp.values)
# y_test = np.log1p(df_test.msrp.values)

# # Remove the target value from the dataframes.
# del df_train['msrp']
# del df_val['msrp']
# del df_test['msrp']
