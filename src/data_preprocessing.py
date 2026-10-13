import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_preprocessor(X):
    numerical_feature = X.select_dtypes(include=["int64","float64"]).columns.tolist()
    categorical_feature = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers = [(
            "num",StandardScaler(),numerical_feature
        ),(
            "cat",OneHotEncoder(handle_unknown="ignore"),categorical_feature
        )]
    )
    return preprocessor