import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def get_data():
    data_dir = '/home/dan/data/cover/'
    P = pd.read_csv(data_dir + 'test.csv', header=None)
    V = pd.read_csv(data_dir + 'val.csv', header=None)
    T = pd.read_csv(data_dir + 'train.csv', header=None)

    # numerical columns
    num_vars = T.columns[:10]
    # all columns but last
    all_vars = T.columns[:-1]

    # standardize
    P.loc[:, num_vars] = (P.loc[:, num_vars] - T[num_vars].mean())/T[num_vars].std()
    V.loc[:, num_vars] = (V.loc[:, num_vars] - T[num_vars].mean())/T[num_vars].std()
    T.loc[:, num_vars] = (T.loc[:, num_vars] - T[num_vars].mean())/T[num_vars].std()

    def convert_to_numpy(df):
        x = df[all_vars].as_matrix()
        y = df[54].as_matrix()
        y = y - 1
        return x, y

    X_test, Y_test = convert_to_numpy(P)
    X_val, Y_val = convert_to_numpy(V)
    X_train, Y_train = convert_to_numpy(T)

    ohe = OneHotEncoder(sparse=False, dtype='float32')
    ohe.fit(Y_train.reshape(-1, 1))
    Y_test = ohe.transform(Y_test.reshape(-1, 1))
    Y_val = ohe.transform(Y_val.reshape(-1, 1))
    Y_train = ohe.transform(Y_train.reshape(-1, 1))

    return X_test, Y_test, X_val, Y_val, X_train, Y_train
