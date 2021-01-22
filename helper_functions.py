"""
necessary functions to process data in the main app
"""
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Ridge

from helper_data import transformers_model

def day_and_time_from_datetime(df):
    """
    this helper function extracts hour, day, and month from the datetime column.

    Params:
    -------
        input: dataframe with the Kaggle bicycle data
        output:new dataframe with the date column transformed to pandas datetime-features
    """
    df = df.copy()
    df['hour'] = pd.to_datetime(df['datetime']).dt.hour
    df['weekday'] = pd.to_datetime(df['datetime']).dt.dayofweek
    df['month'] = pd.to_datetime(df['datetime']).dt.month
    df = df.drop('datetime', axis = 1)
    return df

def build_model(df):
    """builds model used for predictions"""


def user_kind(df, userKind):
    """
    selects y-values accoring to given user
    """
    if userKind == "registered":
        x = df["hour"]
        y = df["registered"]
    elif userKind == "casual":
        x = df["hour"]
        y = df["casual"]
    else:
        x = df["hour"]
        y = df["count"]
    return x, y


def outlier_correction(df, outlierCorrection):
    """
    removes outliers based on given method
    """
    if outlierCorrection == "iqr":
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        df_corrected = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
    elif outlierCorrection == "zscore":
        z = np.abs(stats.zscore(df))
        threshold = 3
        df_corrected = df[(z < 3).all(axis=1)]

    return df_corrected


def day_of_week(df, kindOfDay):
    """
    subsets data to working days or weekends
    """
    if kindOfDay == "weekend":
        df = df.loc[df["workingday"] == 0]
    elif kindOfDay == "weekday":
        df = df.loc[df["workingday"] == 1]
    return df

def select_feature(df, feature_selector):
    """subsets data to individual features"""
    feature = feature_selector
    fr_sub = fr[fr["features"] == feature]
    return fr_sub

DATA = pd.read_csv("./data/train.csv")
df = day_and_time_from_datetime(DATA)

# find outliers by IQR
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df_IQR_corrected = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]

y_all = df_IQR_corrected['count']
X = df_IQR_corrected.drop(columns=['count'])
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y_all, random_state=42)
y = np.log1p(ytrain)

def get_Xtrain_short():
    X = Xtrain[["atemp", "humidity", "workingday", "hour", "weekday", "month" ]]

    return X
X = get_Xtrain_short()


def build_ridge_model():

    preprocessor = ColumnTransformer(transformers_model)
    p = Pipeline(steps=[('preprocessor', preprocessor),
                    ('classifier', Ridge(alpha=2, max_iter=1000))
                    ])

    ridge_model = p.fit(X, y)
    return ridge_model

ridge_model = build_ridge_model()

def make_prediction(input_dict):
    """does the actual prediction"""

    input_data = pd.DataFrame([input_dict])
    prediction = ridge_model.predict(input_data)[0]
    prediction = np.expm1(prediction)

    return prediction
