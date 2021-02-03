import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px

import pandas as pd
import numpy as np

from helper_functions import day_and_time_from_datetime
from helper_functions import user_kind, outlier_correction, day_of_week
from helper_functions import (
    select_feature,
    make_prediction,
    build_ridge_model,
    get_Xtrain_short,
)

from helper_data import feature_options, metric_options
from helper_data import options_weekday, options_months

DATA = pd.read_csv("./data/train.csv")
df = day_and_time_from_datetime(DATA)
X = get_Xtrain_short()

FEATURE_RESULTS = pd.read_csv("./data/feature_results.csv")
REGR_COMP = pd.read_csv("./data/regr_comp_results.csv")

ridge_model = build_ridge_model()
template = "plotly_white"

very_cold = df["atemp"].quantile(0.2)
cold = df["atemp"].quantile(0.4)
moderate = df["atemp"].quantile(0.6)
warm = df["atemp"].quantile(0.8)
hot = df["atemp"].quantile(1.0)

very_dry = df["humidity"].quantile(0.2)
dry = df["humidity"].quantile(0.4)
foggy = df["humidity"].quantile(0.6)
rainy = df["humidity"].quantile(0.8)
very_rainy = df["humidity"].quantile(1.0)


# initialise the app
app = dash.Dash(__name__)


# define the app
app.layout = html.Div(
    [
        # first row:
        html.Div(
            [
                html.H1("Bicycle Ridge"),
                html.H2(
                    "An interactive visualization of Regression Models to predict Capital Bike Share usage"
                ),
                # left side: image & dropdown
                html.Div(
                    [
                        html.Img(
                            id="bike image",
                            height="180px",
                            src="assets/TheRidge_bike.png",
                            style={"border-radius": "20px"},
                        ),
                        html.Label("Kind of User"),
                        dcc.Dropdown(
                            id="userKind",
                            options=[
                                {"label": "All customers", "value": "count"},
                                {"label": "Frequent customers", "value": "registered"},
                                {"label": "Casual customers", "value": "casual"},
                            ],
                            value="count",
                        ),
                        html.Label("Kind of Day"),
                        dcc.Dropdown(
                            id="kindOfDay",
                            options=[
                                {"label": "All days", "value": "all"},
                                {"label": "Weekend", "value": "weekend"},
                                {"label": "Working Day", "value": "weekday"},
                            ],
                            value="all",
                        ),
                        html.Label("Outlier Correction"),
                        dcc.Dropdown(
                            id="outlierCorrection",
                            options=[
                                {"label": "None", "value": "none"},
                                {"label": "IQR", "value": "iqr"},
                                {"label": "Zscore", "value": "zscore"},
                            ],
                            value="none",
                        ),
                    ],
                    className="pretty-container three columns",
                ),
                # right side: Graph
                html.Div(
                    [
                        html.Div(
                            [
                                html.H5("Bicylce usage throughout the day"),
                                dcc.Graph(id="bicycleUsage"),
                            ],
                            className="pretty-container nine columns",
                        ),
                    ]
                ),
            ]
        ),
        # second row:
        html.Div(
            [
                # left side: graph
                html.Div(
                    [
                        html.H5("Impact of individual features"),
                        html.Label("Features"),
                        dcc.Dropdown(
                            id="feature_selector", options=feature_options, value="hour"
                        ),
                        dcc.Graph(
                            id="feature_graph",
                        ),
                    ],
                    className="pretty-container three columns",
                    style={"margin-left": "2.5%"},
                ),
                # right side: graph
                html.Div(
                    [
                        html.H5("Performance of different regressors"),
                        html.Label("Regressor metric"),
                        dcc.Dropdown(
                            id="regressor_metric",
                            options=metric_options,
                            value="mean_test_r2",
                        ),
                        dcc.Graph(id="regressor_graph"),
                    ],
                    className="pretty container nine columns",
                ),
            ]
        ),
        # third row:
        html.Div(
            [
                # left side: dropdown
                html.Div(
                    [
                        html.Div(
                            [
                                html.H5("Choose Datetime"),
                                html.Label("Hour of the day"),
                                dcc.Dropdown(
                                    id="hour",
                                    options=[
                                        {"label": x, "value": x}
                                        for x in df["hour"].unique()
                                    ],
                                    value=11,
                                ),
                                html.Label("Day of Week"),
                                dcc.Dropdown(
                                    id="weekday", options=options_weekday, value=3
                                ),
                                html.Label("Month of the year"),
                                dcc.Dropdown(
                                    id="month",
                                    options=options_months,
                                    value=1,
                                ),
                                html.Label("Public holiday"),
                                dcc.Dropdown(
                                    id="workingday",
                                    options=[
                                        {"label": "Working Day", "value": 1},
                                        {"label": "Free day ", "value": 0},
                                    ],
                                    value=0,
                                ),
                            ],
                            className="pretty container three columns",
                            style={"margin-left": "5%"},
                        ),
                    ]
                ),
                # middle slider
                html.Div(
                    [
                        html.Div(
                            [
                                html.H5("Choose Weather Conditions"),
                                html.Label("Felt temperature"),
                                dcc.Slider(
                                    id="atemp",
                                    min=very_cold,
                                    max=hot,
                                    step=1.0,
                                    marks={
                                        very_cold: "very_cold",
                                        cold: "cold",
                                        moderate: "moderate",
                                        warm: "warm",
                                        hot: "hot",
                                    },
                                    value=15,
                                ),
                                html.Label("Relative humidity"),
                                dcc.Slider(
                                    id="humidity",
                                    min=very_dry,
                                    max=very_rainy,
                                    step=1.0,
                                    marks={
                                        very_dry: "very_dry",
                                        dry: "dry",
                                        foggy: "foggy",
                                        rainy: "rainy",
                                        very_rainy: "very_rainy",
                                    },
                                    value=75,
                                ),
                            ],
                            className="pretty container four columns",
                            style={"margin-left": "4%"},
                        )
                    ]
                ),
                # right prediction output
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3("How many bikes are needed?"),
                                html.H5(
                                    "Ridge-Prediction from 6 IQR-corrected features and with regularization (alpha = 2)"
                                ),
                                html.H1(id="bicycle_prediction"),
                            ],
                            className="pretty container four columns",
                            style={
                                "border-radius": "20px",
                                "background-color": "#B3F3FF",
                                "margin-left": "5%",
                            },
                        )
                    ]
                ),
            ]
        ),
    ]
)

##########################################################

##########################################################


@app.callback(
    Output("bicycleUsage", "figure"),
    [
        Input("userKind", "value"),
        Input("kindOfDay", "value"),
        Input("outlierCorrection", "value"),
    ],
)
def update_bicycleUsage(userKind, kindOfDay, outlierCorrection, df=df):
    """function to render the main graph in row 1"""

    if outlierCorrection == "none":
        if kindOfDay == "all":
            x, y = user_kind(df, userKind)
        else:
            df_day = day_of_week(df, kindOfDay)
            x, y = user_kind(df_day, userKind)
    else:
        df_corrected = outlier_correction(df, outlierCorrection)
        if kindOfDay == "all":
            x, y = user_kind(df_corrected, userKind)
        else:
            df_day = day_of_week(df_corrected, kindOfDay)
            x, y = user_kind(df_day, userKind)

    fig = px.box(
        x=x, y=y, labels={"x": "Hour of the day", "y": "Count"}, template=template
    )
    return fig


@app.callback(Output("feature_graph", "figure"), Input("feature_selector", "value"))
def update_feature_selection(feature_selector):
    """function to render the feature_selection graph in row 2, left"""
    feature = feature_selector
    fr_sub = FEATURE_RESULTS.loc[FEATURE_RESULTS["features"] == feature]
    fr_sub = fr_sub.T
    fr_sub.columns = fr_sub.iloc[0]
    fr_sub.drop("features", inplace=True)
    fr_sub.reset_index(inplace=True)
    y = fr_sub[feature]
    x = fr_sub["index"]

    fig = px.bar(x=x, y=y, labels={"x": "Metric", "y": "Score"}, template=template)
    return fig


@app.callback(Output("regressor_graph", "figure"), Input("regressor_metric", "value"))
def update_regressor_metric(regressor_metric):
    """function to render the regressor_metric graph in row 2, right"""

    metric = regressor_metric
    fig = px.scatter(
        REGR_COMP,
        x=metric,
        y="mean_fit_time",
        size="mean_test_r2",
        color="param_classifier",
        labels={
            "mean_fit_time": "Average fitting time [s]",
            "param_classifier": "Regressor",
        },
        template=template,
    )

    return fig


@app.callback(
    Output(component_id="bicycle_prediction", component_property="children"),
    [
        Input("atemp", "value"),
        Input("humidity", "value"),
        Input("workingday", "value"),
        Input("hour", "value"),
        Input("weekday", "value"),
        Input("month", "value"),
    ],
)
def update_bicycle_prediction(atemp, humidity, workingday, hour, weekday, month):
    """function that outputs accurate predictions"""
    input_dict = {
        "atemp": atemp,
        "humidity": humidity,
        "workingday": workingday,
        "hour": hour,
        "weekday": weekday,
        "month": month,
    }
    input_data = pd.DataFrame([input_dict])
    input_data.index = ["datapoint"]

    prediction = ridge_model.predict(input_data)[0]
    prediction = np.expm1(prediction)

    return "{}".format(int(prediction))


# run the app
if __name__ == "__main__":
    app.run_server(debug=True)
