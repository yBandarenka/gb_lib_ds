import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_auc_score

# Задание 1

# data_url = "http://lib.stat.cmu.edu/datasets/boston"
# raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
# data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
# target = raw_df.values[1::2, 2]

load_data = load_boston()
data_boston = load_data["data"]

feature_names_boston = load_data["feature_names"]
X = pd.DataFrame(data_boston, columns=feature_names_boston)

target_boston = load_data["target"]
Y = pd.DataFrame(target_boston, columns=["price"])

print(X.head())
print(Y.head())

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

linear_regression = LinearRegression()
linear_regression.fit(X_train, Y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

y_model_predict_linear_regression = linear_regression.predict(X_test)
result_test_model_linear = pd.DataFrame({
    "Y_test": Y_test["price"],
    "Y_model_predict_linear": y_model_predict_linear_regression.flatten()})

print(result_test_model_linear.head())

mean_squared_error_model_linear = mean_squared_error(result_test_model_linear["Y_model_predict_linear"],
                                                     result_test_model_linear["Y_test"])

print(mean_squared_error_model_linear)

# Задание 2

model = RandomForestRegressor(n_estimators=1000, max_depth=12, random_state=42)
model.fit(X_train, Y_train.values[:, 0])

RandomForestRegressor(bootstrap=True, criterion='mse',
                      max_depth=12,
                      max_features='auto', max_leaf_nodes=None, min_impurity_decrease=1e-07,
                      min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0,
                      n_estimators=1000,
                      n_jobs=None, oob_score=False,
                      random_state=42,
                      verbose=0, warm_start=False)

y_predict_model = model.predict(X_test)

result_test_model = pd.DataFrame({
    "Y_test": Y_test["price"],
    "Y_predict_model": y_predict_model.flatten()})

print(result_test_model.head())

mean_squared_error_model = mean_squared_error(result_test_model["Y_predict_model"],
                                              result_test_model["Y_test"])

print(mean_squared_error_model)

# 21.517444231177357 9.334548946165196 - случайный лес более точен чем линейная регрессия
print(mean_squared_error_model_linear, mean_squared_error_model)

# Задание 3

print(model.feature_importances_)

feature_importance = pd.DataFrame({'name':X.columns,
                                   'feature_importance': model.feature_importances_},
                                  columns=['feature_importance', 'name'])

print(feature_importance)

#LSTAT and RM
print(feature_importance.nlargest(2, 'feature_importance'))

# Задание 4

df = pd.read_csv('creditcard.csv')
info_df = df['Class'].value_counts(normalize=True)

print(info_df)
print(df.info())

pd.options.display.max_columns = 100

print(df.head(10))

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, stratify=y)

print('X_train ', X_train.shape)
print('X_test ', X_test.shape)
print('y_train ', y_train.shape)
print('y_test ', y_test.shape)

parameters = [{
    'n_estimators': [10, 15],
    'max_features': np.arange(3, 5),
    'max_depth': np.arange(4, 7)
}]

grid_search_cv = GridSearchCV(
    estimator=RandomForestClassifier(random_state=100),
    param_grid=parameters,
    scoring='roc_auc',
    cv=3,
)

grid_search_cv.fit(X_train, y_train)

GridSearchCV(cv=3, error_score='raise-deprecating',
             estimator=RandomForestClassifier(bootstrap=True, class_weight=None,
                                              criterion='gini', max_depth=None,
                                              max_features='auto',
                                              max_leaf_nodes=None,
                                              min_impurity_decrease=0.0,
                                              min_samples_leaf=1,
                                              min_samples_split=2,
                                              min_weight_fraction_leaf=0.0,
                                              n_estimators='warn', n_jobs=None,
                                              oob_score=False, random_state=100,
                                              verbose=0, warm_start=False),
             n_jobs=None,
             param_grid=[{'max_depth': np.array([4, 5, 6]),
                          'max_features': np.array([3, 4]),
                          'n_estimators': [10, 15]}],
             pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
             scoring='roc_auc', verbose=0)

print(grid_search_cv.best_params_)

random_forest_classifier = RandomForestClassifier(max_depth=6, max_features=3, n_estimators=15)
random_forest_classifier.fit(X_train, y_train)

RandomForestClassifier(bootstrap=True, class_weight=None,
                       criterion='gini', max_depth=6, max_features=3,
                       max_leaf_nodes=None,
                       min_impurity_decrease=0.0,
                       min_samples_leaf=1,
                       min_samples_split=2,
                       min_weight_fraction_leaf=0.0,
                       n_estimators=15,n_jobs=None,
                       oob_score=False, random_state=None,
                       verbose=0, warm_start=False)

y_pred = random_forest_classifier.predict_proba(X_test)
y_pred_proba = y_pred[:, 1]

result_roc = roc_auc_score(y_test, y_pred_proba)
print(result_roc)

