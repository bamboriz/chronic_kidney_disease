import pandas as pd
import numpy as np
import logging

from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

DATA_URL = './data/data.csv'
CATEGORY_THRESHOLD = 10

def load_data(filter: str = None, filter_value: str = None) -> pd.DataFrame:
    '''
    Load data from DATA_URL
    '''
    df = pd.read_csv(DATA_URL, index_col=0)
    if filter:
        df = df[df[filter] == filter_value]
    return df

def get_categorical_cols() -> [str]:
    '''
    Dynamically detects column as categorical if number of unique values is less than CATEGORY_THRESHOLD
    '''
    df = load_data()
    cols = []
    for each in df.columns:
        if df[each].nunique() <= CATEGORY_THRESHOLD:
            cols.append(each)
    return cols

def get_numeric_cols() -> [str]:
    '''
    Dynamically detect column as numerical if number of unique values is greater than CATEGORY_THRESHOLD
    '''
    df = load_data()
    cols = []
    for each in df.columns:
        if df[each].nunique() > CATEGORY_THRESHOLD:
            cols.append(each)
    return cols

def convert_numeric_cols_to_float(df: pd.DataFrame) -> pd.DataFrame:
    continuous_columns = get_numeric_cols()
    for col in continuous_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def scale_data(X: pd.DataFrame) -> pd.DataFrame:
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def preprocess(X: pd.DataFrame) -> pd.DataFrame:
    '''
    Preprocessing pipeline does missing value imputation and data normalisation
    '''
    X_preprocessed = preprocess_continuous_data(X)
    X_preprocessed = preprocess_categorical_data(X_preprocessed)
    X_preprocessed = scale_data(X_preprocessed)
    return X_preprocessed

def preprocess_continuous_data(X: pd.DataFrame) -> pd.DataFrame:
    X_preprocessed = X.copy()
    continuous_columns = get_numeric_cols()
    X_preprocessed = convert_numeric_cols_to_float(X_preprocessed)
    print(continuous_columns)
    for column_name in continuous_columns:
        print(column_name)
        print(X[column_name].dtypes)
        X_preprocessed[column_name] = X_preprocessed[column_name].fillna(X_preprocessed[column_name].mean())
    return X_preprocessed

def preprocess_categorical_data(X: pd.DataFrame) -> pd.DataFrame:
    X_preprocessed = impute_missing_categorical_data(X)
    X_with_encoded_categorical_features = encode_categorical_features_orchestrator(X_preprocessed)
    return X_with_encoded_categorical_features

def impute_missing_categorical_data(X: pd.DataFrame) -> pd.DataFrame:
    X_preprocessed = X.copy()
    categorical_columns = get_categorical_cols()
    for column_name in categorical_columns:
        X_preprocessed[column_name] = X_preprocessed[column_name].fillna(X_preprocessed[column_name].mode()[0])
    return X_preprocessed

def encode_categorical_features_orchestrator(X: pd.DataFrame) -> pd.DataFrame:
    one_hot_encoder = get_categorical_encoder(X)
    X_with_continuous_data_and_encoded_categorical_data = encode_categorical_features(one_hot_encoder, X)
    return X_with_continuous_data_and_encoded_categorical_data

def get_categorical_encoder(X: pd.DataFrame):
    logging.info('Generating a new encoder')
    categorical_columns = get_categorical_cols()
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore', dtype=int, sparse=True)
    one_hot_encoder.fit(X[categorical_columns])
    return one_hot_encoder

def encode_categorical_features(one_hot_encoder, X):
    categorical_columns = get_categorical_cols()
    continuous_columns = get_numeric_cols()
    encoded_categorical_data_matrix = one_hot_encoder.transform(X[categorical_columns])
    encoded_data_columns = one_hot_encoder.get_feature_names(categorical_columns)
    encoded_categorical_data_df = pd.DataFrame.sparse.from_spmatrix(data=encoded_categorical_data_matrix,
                                                                    columns=encoded_data_columns, index=X.index)
    X_with_continuous_data_and_encoded_categorical_data = X.copy()[continuous_columns].join(encoded_categorical_data_df)
    return X_with_continuous_data_and_encoded_categorical_data

def get_inertia(X: np.ndarray, n: int) -> tuple:
    distortions = []
    x = range(2, n)
    for k in range(2, 11):
        KMeans_model = KMeans(n_clusters=k, random_state=42)
        KMeans_model.fit(X)
        distortions.append(KMeans_model.inertia_)
    return (x, distortions)