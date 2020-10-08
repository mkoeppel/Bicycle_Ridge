#!/usr/bin/env python
# coding: utf-8

# # Linear Regression Modeling using Kaggles CapitalBikeShare dataset
#
# ## explanatory data analysis and linear regression modeling using scikit-learn


import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as mpt

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, PassiveAggressiveRegressor
from sklearn.linear_model import  RidgeCV, LassoCV, ElasticNetCV

from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def day_and_time_from_datetime(df):
    '''
    this helper function extracts hour, day, and month from the datetime column.

    Params:
    -------
        input: dataframe with the Kaggle bicycle data
        output:new dataframe with the date column transformed to pandas datetime-features
    '''
    df = df.copy()
    df['time [h]'] = pd.to_datetime(df['datetime']).dt.hour
    df['weekday'] = pd.to_datetime(df['datetime']).dt.dayofweek
    df['month'] = pd.to_datetime(df['datetime']).dt.month
    df = df.drop('datetime', axis = 1)
    return df


df = pd.read_csv("./data/train.csv")
y = df['count']
X = df.drop(columns= ['count'])


Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, random_state = 42)


sns.heatmap(Xtrain.isna())
plt.title('detecting missing values')
plt.show(block = False)


df = day_and_time_from_datetime(df)

def usage_by_customer(df):
    plt.figure(figsize = (15,5))
    sns.lineplot(x = 'time [h]', y =  'count', data = df, label = 'all customers')
    sns.lineplot(x = 'time [h]', y =  'registered', data = df, label = 'registered customers')
    sns.lineplot(x = 'time [h]', y =  'casual', data = df, label = 'casual customers')
    plt.title('differences in bicyle usage throughout the day')
    plt.show(block = False)

usage_by_customer(df)


def usage_by_day(df):
    plt.figure(figsize = (15,5))
    orange_patch = mpt.Patch(color = sns.color_palette()[1], label = 'workingday')
    blue_patch = mpt.Patch(color = sns.color_palette()[0], label = 'weekend')

    sns.lineplot( x = 'time [h]', y = 'count', hue = 'workingday', data = df)
    plt.legend(labels =  ['weekend', 'workingday'])
    plt.title('all users bicycle usage at weekdays and weekends')

    f, (ax1, ax2) = plt.subplots(1, 2, figsize = (15,5))
    sns.lineplot( x = 'time [h]', y = 'registered', hue = 'workingday', data = df, ax = ax1)
    sns.lineplot(x = 'time [h]', y = 'casual', hue = 'workingday', data = df, ax = ax2)
    ax1.set_title('bicycle usage from registered users')
    ax2.set_title('bicycle usage from casual users')
    ax1.legend(handles = [ blue_patch, orange_patch])
    ax2.legend(handles = [ blue_patch, orange_patch])

    plt.show(block = False)

def usage_by_day(df):

    f, (ax1, ax2) = plt.subplots(1, 2, figsize = (15,5))
    orange_patch = mpt.Patch(color = sns.color_palette()[1], label = 'workingday')
    blue_patch = mpt.Patch(color = sns.color_palette()[0], label = 'weekend')

    sns.violinplot(x = 'weather', y = 'count', data = df, hue = 'workingday', split = True, ax = ax1)
    sns.violinplot(x = 'season', y = 'count', data = df, hue = 'workingday', split = True, ax = ax2)

    ax1.set_title('bicycle usage depending on the weather')
    ax2.set_title('bicycle usage depending on the season')

    ax1.legend(handles = [ blue_patch, orange_patch])
    ax2.legend(handles = [ blue_patch, orange_patch])
    plt.show(block = False)

usage_by_day(df)


def usage_weather(df):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize = (15,5))
    sns.regplot(x = 'temp', y = 'count', data = df, ax = ax1, line_kws={"color": sns.color_palette()[1]})
    sns.regplot(x = 'humidity', y = 'count', data = df, ax = ax2, line_kws={"color": sns.color_palette()[1]})
    ax1.set_title('correlation between bicycle usage and temperature')
    ax2.set_title('correlation between bicycle usage and humidity')

usage_weather(df)


# features from the original datetime-column are extracted, onehot-encoded and defined as interaction-features
time_feat = ['datetime']
time_transformer = Pipeline(steps=[
    ('day_time_extract', FunctionTransformer(day_and_time_from_datetime)),
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ('polynomial', PolynomialFeatures(interaction_only=True, include_bias=False))
    ])

# the numeric parameters windspeed and atemp are scaled to a mean-value of 0 and a variance of 1
numeric_features = ['windspeed', 'atemp']
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ])

# categorical feature weather is onehot-encoded
categorical_features = ['weather']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

# three more interacting features are defined: (humidity and workingday), (workingday and temp), and (workingday and season), all are onehot-encoded afterwards
interact_features1 = ['humidity' ,'workingday']
interaction_transformer1 = Pipeline(steps=[
    ('polynomial1', PolynomialFeatures(interaction_only = True, include_bias=False)),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

interact_features2 = ['workingday' ,'temp']
interaction_transformer2 = Pipeline(steps=[
    ('polynomial2', PolynomialFeatures(interaction_only = True, include_bias=False)),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

interact_features3 = ['workingday' ,'season']
interaction_transformer3 = Pipeline(steps=[
    ('polynomial3', PolynomialFeatures(interaction_only = True, include_bias=False)),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])


preprocessor = ColumnTransformer(
    transformers=[
        ('day_time_extract', time_transformer, time_feat),
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('polynomial1', interaction_transformer1, interact_features1),
        ('polynomial2', interaction_transformer2, interact_features2),
        ('polynomial3', interaction_transformer3, interact_features3)
        ])


p = Pipeline(steps=[('preprocessor', preprocessor),
                    ('classifier', LinearRegression())
                    ])


p.fit(Xtrain, ytrain)
ypred_train = p.predict(Xtrain)
print(f'MSE from LinReg: ' + str(mean_squared_error(ytrain, ypred_train)))
print('MAE from LinReg: ' + str(mean_absolute_error(ytrain, ypred_train)))
print('r2 from LinReg: ' + str(r2_score(ytrain, ypred_train)))

print(f'starting with grid search')

param_grid_reg = {
    'classifier' : [LinearRegression(), Ridge(), Lasso(), ElasticNet(), LinearSVR(), SGDRegressor(), PassiveAggressiveRegressor(), KNeighborsRegressor(), DecisionTreeRegressor(),  GradientBoostingRegressor()]
}

grid_search_reg = GridSearchCV(p, param_grid_reg)
grid_search_reg.fit(Xtrain, ytrain)

def plot_results_grid_search_reg(GS_results):
    GS_reg = pd.DataFrame(grid_search_reg.cv_results_)
    GS_reg = GS_reg.stack().unstack(0).reset_index()
    GS_reg.set_index('index', inplace = True)
    GS_reg.columns = param_grid_reg['classifier']
    GS_reg.drop(['param_classifier','params'], inplace = True)

    f, (ax1, ax2) = plt.subplots(2, 1, figsize = (15,5))
    ax1.set_title('performance of different linear regressors')

    sns.barplot(y = GS_reg.iloc[0], x = GS_reg.columns, ax = ax2)
    sns.barplot(y = GS_reg.iloc[9], x = GS_reg.columns, ax = ax1)
    ax1.set_xticks([])
    ax1.set_ylabel('mean r2-score')
    ax2.set_ylabel('mean fitting time')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    plt.show(block = False)

plot_results_grid_search_reg(grid_search_reg)


print(f'start with grid search cross-validation for regularization methods')
param_grid_CV = {
    'classifier' : [ RidgeCV(), LassoCV(), ElasticNetCV()]
}

grid_search_CV = GridSearchCV(p, param_grid_CV)
grid_search_CV.fit(Xtrain, ytrain)

def plot_grid_search_cross_val(GS_results):
    GS_CV = pd.DataFrame(grid_search_CV.cv_results_)
    GS_score = GS_CV.stack().unstack(0).reset_index().iloc[6:11]
    GS_score.rename(columns = {'index' :'index',  0: 'RidgeCV', 1: 'LassoCV', 2:'ElasticNetCV'}, inplace =True)
    GS_t = pd.DataFrame(GS_CV.stack().unstack(0).reset_index().iloc[0])
    GS_time = GS_t.T
    GS_time.rename(columns = {'index' :'index',  0: 'RidgeCV', 1: 'LassoCV', 2:'ElasticNetCV'}, inplace =True)
    GS_time.set_index('index', inplace = True)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize = (15,5))
    sns.barplot(data = GS_score, ax = ax1)
    sns.barplot(data = GS_time, ax = ax2)
    ax1.set_title('r2-score model cross validation')
    ax2.set_title('mean fitting time cross validation')
    plt.show()
plot_grid_search_cross_val(grid_search_CV)
