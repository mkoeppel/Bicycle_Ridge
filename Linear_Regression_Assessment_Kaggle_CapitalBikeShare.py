#!/usr/bin/env python
# coding: utf-8

"""
Linear Regression Modeling using Kaggles CapitalBikeShare dataset
explanatory data analysis and linear regression modeling using pandas/ seaborn and scikit-learn
"""

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import patches as mpt


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
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
    df = df.drop('datetime', axis=1)
    return df


df = pd.read_csv("./data/train.csv")
y_counts = df['count']
X_variables = df.drop(columns=['count'])
df.head()


Xtrain, Xtest, ytrain, ytest = train_test_split(X_variables, y_counts, random_state=42)


sns.heatmap(Xtrain.isna())
plt.title('detecting missing values')
plt.show(block=False)

df = day_and_time_from_datetime(df)

def usage_by_customer(df):
    """"
    plots the bicylce demand throughout the day by different customers
    """
    plt.figure(figsize=(15, 5))
    sns.lineplot(x='time [h]', y='count', data=df, label='all customers')
    sns.lineplot(x='time [h]', y='registered', data=df, label='registered customers')
    sns.lineplot(x='time [h]', y='casual', data=df, label='casual customers')
    plt.title('differences in bicyle usage throughout the day')
    plt.show(block=False)

usage_by_customer(df)

def usage_by_day(df):
    """
    plots bicyle demand on weekends and working days, split by customers also
    """
    plt.figure(figsize=(15, 5))
    orange_patch = mpt.Patch(color=sns.color_palette()[1], label='workingday')
    blue_patch = mpt.Patch(color=sns.color_palette()[0], label='weekend')

    sns.lineplot(x='time [h]', y='count', hue='workingday', data=df)
    plt.legend(labels=['weekend', 'workingday'])
    plt.title('all users bicycle usage at weekdays and weekends')

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    sns.lineplot(x='time [h]', y='registered', hue='workingday', data=df, ax=ax1)
    sns.lineplot(x='time [h]', y='casual', hue='workingday', data=df, ax=ax2)
    ax1.set_title('bicycle usage from registered users')
    ax2.set_title('bicycle usage from casual users')
    ax1.legend(handles=[blue_patch, orange_patch])
    ax2.legend(handles=[blue_patch, orange_patch])

    plt.show(block=False)


def usage_by_season(df):
    """
    plots bicyle demand depeding on season and weater conditions
    """
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    orange_patch = mpt.Patch(color=sns.color_palette()[1], label='workingday')
    blue_patch = mpt.Patch(color=sns.color_palette()[0], label='weekend')

    sns.violinplot(x='weather', y='count', data=df, hue='workingday', split=True, ax=ax1)
    sns.violinplot(x='season', y='count', data=df, hue='workingday', split=True, ax=ax2)

    ax1.set_title('bicycle usage depending on the weather')
    ax2.set_title('bicycle usage depending on the season')

    ax1.legend(handles=[blue_patch, orange_patch])
    ax2.legend(handles=[blue_patch, orange_patch])
    plt.show(block=False)

usage_by_season(df)

def usage_by_weather(df):
    """
    plot correlates bicyle demand with temperature, windspeed and humidity
    """
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    sns.regplot(x='temp', y='count', data=df, ax=ax1, line_kws={"color": sns.color_palette()[1]})
    sns.regplot(x='humidity', y='count', data=df, ax=ax2, line_kws={"color": sns.color_palette()[1]})
    sns.regplot(x='windspeed', y='count', data=df, ax=ax3, line_kws={"color": sns.color_palette()[1]})

    ax1.set_title('correlation between bicycle usage and temperature')
    ax2.set_title('correlation between bicycle usage and humidity')
    ax3.set_title('correlation between bicycle usage and windspeed')
    plt.show(block=False)

usage_by_weather(df)


# features from the original datetime-column are extracted, onehot-encoded and defined as interaction-features
time_feat = ['datetime']
time_transformer = Pipeline(steps=[
    ('day_time_extract', FunctionTransformer(day_and_time_from_datetime)),
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ('polynomial', PolynomialFeatures(interaction_only=True, include_bias=False))
    ])

#the numeric parameters windspeed and temp are scaled to a mean-value of 0 and a variance of 1
numeric_features = ['temp']
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
    ])

binned_features = ['windspeed']
binned_transformer = Pipeline(steps=[
    ('binner', KBinsDiscretizer(n_bins=8, encode='onehot', strategy='quantile'))
    ])

# categorical feature weather is onehot-encoded
categorical_features = ['weather', 'season']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

# three more interacting features are defined: (humidity and workingday), (workingday and temp), and (workingday and season), all are onehot-encoded afterwards
interact_features1 = ['workingday', 'humidity']
interaction_transformer1 = Pipeline(steps=[
    ('polynomial1', PolynomialFeatures(interaction_only=True, include_bias=False)),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

interact_features2 = ['workingday', 'temp']
interaction_transformer2 = Pipeline(steps=[
    ('polynomial2', PolynomialFeatures(interaction_only=True, include_bias=False)),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

interact_features3 = ['workingday', 'season']
interaction_transformer3 = Pipeline(steps=[
    ('polynomial3', PolynomialFeatures(interaction_only=True, include_bias=False)),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

interact_features4 = ['workingday', 'windspeed']
interaction_transformer4 = Pipeline(steps=[
    ('polynomial4', PolynomialFeatures(interaction_only=True, include_bias=False)),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])


preprocessor = ColumnTransformer(
    transformers=[
        ('day_time_extract', time_transformer, time_feat),
        ('num', numeric_transformer, numeric_features),
        ('bin', binned_transformer, binned_features),
        ('cat', categorical_transformer, categorical_features),
        ('polynomial1', interaction_transformer1, interact_features1),
        ('polynomial2', interaction_transformer2, interact_features2),
        ('polynomial3', interaction_transformer3, interact_features3),
        ('polynomial4', interaction_transformer4, interact_features4)
        ])


p = Pipeline(steps=[('preprocessor', preprocessor),
                    ('classifier', LinearRegression())
                    ])



p.fit(Xtrain, ytrain)
ypred_train = p.predict(Xtrain)
print('MSE from LinReg: ' + str(mean_squared_error(ytrain, ypred_train)))
print('MAE from LinReg: ' + str(mean_absolute_error(ytrain, ypred_train)))
print('r2 from LinReg: ' + str(r2_score(ytrain, ypred_train)))


print(f'starting with grid search')

# In[16]:


param_grid_reg = {
    'classifier' : [LinearRegression(), Ridge(), Lasso(), ElasticNet(), LinearSVR(), SGDRegressor(), PassiveAggressiveRegressor(), KNeighborsRegressor(), DecisionTreeRegressor(), GradientBoostingRegressor()]
}
grid_search_reg = GridSearchCV(p, param_grid_reg)


grid_search_reg.fit(Xtrain, ytrain)

def plot_results_grid_search_reg(GS_results):
    """
    process and plot results from grid_search
    """
    GS_reg = pd.DataFrame(grid_search_reg.cv_results_)
    GS_reg = GS_reg.stack().unstack(0).reset_index()
    GS_reg.set_index('index', inplace=True)
    GS_reg.columns = param_grid_reg['classifier']
    GS_reg.drop(['param_classifier', 'params'], inplace=True)

    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 5))
    ax1.set_title('performance of different linear regressors')

    sns.barplot(y=GS_reg.iloc[0], x=GS_reg.columns, ax=ax2)
    sns.barplot(y=GS_reg.iloc[9], x=GS_reg.columns, ax=ax1)
    ax1.set_xticks([])
    ax1.set_ylabel('mean r2-score')
    ax2.set_ylabel('mean fitting time')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    plt.show(block=False)

plot_results_grid_search_reg(grid_search_reg)


print(f'start with grid search cross-validation for regularization methods')


param_grid_CV = {
    'classifier' : [RidgeCV(), LassoCV(), ElasticNetCV()]
}
grid_search_CV = GridSearchCV(p, param_grid_CV)



grid_search_CV.fit(Xtrain, ytrain)

def plot_grid_search_cross_val(GS_results):
    """
    plot results from hyperparameter_optimization
    """
    GS_CV = pd.DataFrame(grid_search_CV.cv_results_)
    GS_score = GS_CV.stack().unstack(0).reset_index().iloc[6:11]
    GS_score.rename(columns={'index' :'index', 0: 'RidgeCV', 1: 'LassoCV', 2: 'ElasticNetCV'}, inplace=True)
    GS_t = pd.DataFrame(GS_CV.stack().unstack(0).reset_index().iloc[0])
    GS_time = GS_t.T
    GS_time.rename(columns={'index' :'index', 0: 'RidgeCV', 1: 'LassoCV', 2: 'ElasticNetCV'}, inplace=True)
    GS_time.set_index('index', inplace=True)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    sns.barplot(data=GS_score, ax=ax1)
    sns.barplot(data=GS_time, ax=ax2)
    ax1.set_title('r2-score model cross validation')
    ax2.set_title('mean fitting time cross validation')
    plt.show(block=False)

plot_grid_search_cross_val(grid_search_CV)



print(f'start with predictions and comparison with ytrue')


ypred = grid_search_CV.predict(Xtrain)

print('MSE from grid_search_CV: ' + str(mean_squared_error(ytrain, ypred)))
print('MAE from grid_search_CV: ' + str(mean_absolute_error(ytrain, ypred)))
print('r2 from grid_search_CV: ' + str(r2_score(ytrain, ypred)))



sns.scatterplot(ytrain, ypred)
plt.title('correlation of real and predicted count data')


Xtrain['hour'] = pd.to_datetime(Xtrain['datetime']).dt.hour
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
sns.lineplot(x=Xtrain['hour'], y=ypred, ax=ax1)
sns.lineplot(x=Xtrain['hour'], y=ytrain, ax=ax2, color=sns.color_palette()[1])
ax1.set_title('predicted bicycle demand per hour')
ax1.set_ylabel('count')
ax1.set_xlabel('hour of the day')
ax2.set_xlabel('hour of the day')
ax1.set_ylim(0, 500)
ax2.set_title('real bicycle demand per hour')



ypred_test = grid_search_CV.predict(Xtest)



print('MSE from grid_search_CV: ' + str(mean_squared_error(ytest, ypred_test)))
print('MAE from grid_search_CV: ' + str(mean_absolute_error(ytest, ypred_test)))
print('r2 from grid_search_CV: ' + str(r2_score(ytest, ypred_test)))




sns.scatterplot(ytest, ypred_test)
plt.title('correlation of real and predicted count data')


Xtest['hour'] = pd.to_datetime(Xtest['datetime']).dt.hour
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
sns.lineplot(x=Xtest['hour'], y=ypred_test, ax=ax1)
sns.lineplot(x=Xtest['hour'], y=ytest, ax=ax2)
ax1.set_title('predicted bicycle demand per hour')
ax2.set_title('real bicycle demand per hour')
