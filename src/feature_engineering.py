from sklearn.base import BaseEstimator,TransformerMixin
import pandas as pd

def create_features(df):
    df=df.copy()
    df["AverageMonthlySpend"] = (df["TotalCharges"] / df["tenure"].replace(0, 1))

    df["IsLongTermContract"] = (df["Contract"].isin(["One year", "Two year"]).astype(int)
    )

    df["HasInternetService"] = (df["InternetService"] != "No").astype(int)
    
    return df

class FeatureEngineer(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        return create_features(X)