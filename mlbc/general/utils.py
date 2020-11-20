import numbers

import pandas as pd
import numpy as np


def read_data(file):
    """Read a dataframe from a CSV file.

    Parameters:
    file (string): path to a CSV file

    Returns:
    DataFrame holding the contents of the file.
    """
    df = pd.read_csv(file)
    return df


def clean_alphanum_data(df):
    """Clean up alphanumeric data in a dataframe.

    Convert all strings to lower case and replace spaces with underscores.

    Parameters:
    df (DataFrame): the dataframe to be cleaned.

    Returns:
    None.
    """
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    string_columns = list(df.dtypes[df.dtypes == 'object'].index)
    for col in string_columns:
        df[col] = df[col].str.lower().str.replace(' ', '_')


def split_data_frame(df, split=0.2, seed=None):
    """Split a dataframe into a train, validation and test set.

    The dataframe is first randomized and then split into three parts.

    Parameters:
    df (DataFrame): the dataframe to split.
    split (float):  fraction of the dataframe to use for validation and test sets.
    seed (int):     the seed used for randomization.

    Returns:
    3-tuple of DataFrame, DataFrame, DataFrame (train, validation, test).
    """
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


def binary_encode(df, feature, n=5):
    """Binary encode a categorical feature.

    Take the top n values of feature and add features to df to binary encode
    feature.  The dataframe is modified in place.

    Parameters:
    df (DataFrame): the dataframe to add the feature to.
    feature (string): feature in df to be binary encoded.
    n (int): number of values for feature to encode.

    Returns:
    List of new features.
    """
    assert feature in df

    top_values = df[feature].value_counts().head(n)
    new_features = []
    for v in top_values.keys():
        binary_feature = feature + '_%s' % v
        df[binary_feature] = (df[feature] == v).astype(int)
        new_features.append(binary_feature)

    return new_features


def encode_age(df, year_field, current_year):
    """Encode the age of an item as a feature.

    The age is calculated on the basis of the contents of `year_field` and
    `current_year`.

    Parameters:
    df (DataFrame): dataframe to encode the age in.
    year_feature (string): the feature that encodes the relevant year.
    current_year (int): the year used to calculate the age.

    Returns:
    Constant value ['age'].
    """
    assert year_field in df
    assert df[year_field].dtype == 'int64'

    df['age'] = current_year - df[year_field]

    return ['age']


def prepare_X(df, base, fns=[]):
    """Prepare a dataframe for learning.

    Convert the dataframe to a Numpy array:

    - Extract the features in `base`.

    - Apply the functions in `fns` to the dataframe to derive new features from
      existing ones (e.g., for binary encoding).

      The elements of `fns` should be tuples `(fn, list_of_args)`. Before
      calling each function, `df` is prepended to the list of arguments. The
      return value should be a list of names of the new feature(s).

    - Fill any missing data with 0.

    Note that `df` is not modified.

    Parameters:
    df (DataFrame): dataframe to convert.
    base (list of strings): list of fields in the dataframe to be used for the array.
    fns (list of tuples (function, arg list)): feature engineering functions.

    Returns:
    ndarray of the prepared data.
    """
    df = df.copy()
    features = base.copy()

    for fn, args in fns:
        args = [df] + args
        new_features = fn(*args) # Note: this should also modify the local copy of `df`!
        features += new_features

    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values

    return X


def linear_regression(X, y, r=0.0):
    """Perform linear regression.

    Parameters:
    X (ndarray): array of input values.
    y (ndarray): target values.
    r (float): regularization amount.

    Returns:
    Tuple of float, ndarray (bias, array of weights)
    """
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    reg = r * np.eye(XTX.shape[0])
    XTX = XTX + reg

    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)

    return w[0], w[1:]


def rmse(y, y_pred):
    """Compute the root mean square error.

    Parameters:
    y (ndarray): target values.
    y_pred (ndarray): predicted values.

    Returns:
    float
    """
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
