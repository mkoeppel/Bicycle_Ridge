# additional data and list-options for the main app

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline


feature_options = [
    {"label": "Hour", "value": "hour"},
    {"label": "Weekday", "value": "weekday"},
    {"label": "Month", "value": "month"},
    {"label": "Felt temperature", "value": "atemp"},
    {"label": "Humidity", "value": "humidity"},
    {"label": "Windspeed", "value": "windspeed"},
    {"label": "Season", "value": "season"},
    {"label": "Weather", "value": "weather"},
    {"label": "Workingday", "value": "workingday"},
    {"label": "Holiday", "value": "holiday"},
    {"label": "Hour x Weekday x Month", "value": "hourweekdaymonth"},
    {"label": "Humidity x Felt temp", "value": "atemphumidity"},
]

metric_options = [
        {"label": "r2 test score", "value": "mean_test_r2"},
        {"label": "r2 train score", "value": "mean_train_r2"},
        {"label": "Test RMSE", "value":  "test_root_mean_squared_error"},
        {"label": "Train RMSE", "value": "train_root_mean_squared_error"},
        {"label": "Test MAE", "value": "test_mean_absolute_error"},
        {"label": "Train MAE", "value": "train_mean_absolute_error"},
        ]

options_months = [
        {"label": "January", "value": 1},
        {"label": "February", "value": 2},
        {"label": "March", "value": 3},
        {"label": "April", "value": 4},
        {"label": "May", "value": 5},
        {"label": "June", "value": 6},
        {"label": "July", "value": 7},
        {"label": "August", "value": 8},
        {"label": "September", "value": 9},
        {"label": "October", "value": 10},
        {"label": "November", "value": 11},
        {"label": "December", "value": 12}]

options_weekday = [
               {"label": "Monday", "value": 0},
               {"label": "Tuesday", "value": 1},
               {"label": "Wednesday", "value": 2},
               {"label": "Thursday", "value": 3},
               {"label": "Friday", "value": 4},
               {"label": "Saturday", "value": 5},
               {"label": "Sunday", "value": 6}
               ]


binned_features1 = ['atemp']
binned_transformer1 = Pipeline(steps=[
    ('binner', KBinsDiscretizer(n_bins = 6, encode = 'onehot', strategy = 'quantile'))
    ])

binned_features2 = ['humidity']
binned_transformer2 = Pipeline(steps=[
    ('binner', KBinsDiscretizer(n_bins = 6, encode = 'onehot', strategy = 'quantile'))
    ])

categorical_features1 = ['workingday']
categorical_transformer1 = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])


interact_features1 = ['hour', 'weekday', 'month']
interaction_transformer1 = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ('polynomial1', PolynomialFeatures(interaction_only = True, include_bias=False)),
    ])

interact_features2 = ['atemp', 'humidity']
interaction_transformer2 = Pipeline(steps=[
    ('binner', KBinsDiscretizer(n_bins = 6, encode = 'onehot', strategy = 'quantile')),
    ('polynomial2', PolynomialFeatures(interaction_only = True, include_bias=False)),
    ])

transformers_model=[
    ('bin1', binned_transformer1, binned_features1),
    ('bin2', binned_transformer2, binned_features2),
    ('cat1', categorical_transformer1, categorical_features1),
    ('polynomial1', interaction_transformer1, interact_features1),
    ('polynomial2', interaction_transformer2, interact_features2),
    ]
