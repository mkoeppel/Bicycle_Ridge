# LinearRegression_CapitalBikeShare

This project is about the data exploration from Kaggles Capital Bike Share dataset from Washington D.C., followed by the the evaluation of various Linear Regressors from the Scikit-learn universe. Key results are displayed in an interactive dashboard.

### tech used:

![alt text](https://github.com/mkoeppel/Bicycle_Ridge/blob/main/Tech_stack_bicycle_ridge.png)

Trends and features that might be useful for feature engineering indicate that: \
  frequent 'registered users' behave differently compared to occasional 'casual' users, with the former using bicycles mainly during workingdays and peaking in the morning and afternoon, while the latter having a clear preference for the time around noon on weekends. 
  weather conditions have a clear impact on bicycle usage, as higher temperature and lower humidity increase bike demand.

(dataset can be found here: https://www.kaggle.com/c/bike-sharing-demand/data.) \ 

In order to build a linear model systematic feature engineering was performed using datetime and weather information. Engineered features are processed through a sklearn-pipeline, which subsequently included Kfold cross-validation. Testing different regressors resulted in the usage of RidgeRegression, as it showed best performance with the given features, but also showed the highest fitting speed, compared to all similarily well performing regressors:

#### r2: 82.6% 
#### RMSE: 54.9
#### Mean Fit Time: 0.09s


Overview about dashboard: 

![alt text](https://github.com/mkoeppel/Bicycle_Ridge/blob/main/bicycle_ridge.gif)


The jupyter notebook goes through the process step by step, also generating two additional files necessary for running the dashboard and giving some static overviwe about model performances:

![alt text](https://github.com/mkoeppel/Bicycle_Ridge/blob/main/bicycle_demand.png)

The selection of Regressors to test was inspired from this excellent article by Q. Lanners
https://towardsdatascience.com/choosing-a-scikit-learn-linear-regression-algorithm-dd96b48105f5


## to do:
~~interactive visualization with plotly/dash~~ \
automate feature selection
