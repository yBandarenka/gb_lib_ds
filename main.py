
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm
from scipy import stats

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split


#

#преобразование типов -> оптимизация памяти
def opt_data_frame(df_in):

#   data_frame = df_in

    for column in df_in.columns:
        if df_in[column].dtypes.kind == 'i' or df_in[column].dtypes.kind == 'u':
            if df_in[column].min() >= 0:
                df_in[column] = pd.to_numeric(df_in[column], downcast='unsigned')
            else:
                df_in[column] = pd.to_numeric(df_in[column], downcast='integer')

        elif df_in[column].dtypes.kind == 'f' or df_in[column].dtypes.kind == 'c':
            df_in[column] = pd.to_numeric(df_in[column], downcast='float')

        elif df_in[column].dtypes.kind == 'O':
            num_unique_values = len(df_in[column].unique())
            num_total_values = len(df_in[column])
            if num_unique_values / num_total_values < 0.5:
                df_in[column] = df_in[column].astype('category')

    return df_in

def fix_room_data_frame(df_in):
    info_by_district_id = df_in.groupby(['DistrictId', 'HouseYear'], as_index=False)\
        .agg({'Rooms': 'sum', 'Square': 'sum'})\
        .rename(columns={'Rooms': 'sum_roos_dr', 'Square': 'sum_square_dr'})

    info_by_district_id['mean_square_per_room_in_dr'] = info_by_district_id['sum_square_dr'] / info_by_district_id['sum_roos_dr']
    info_by_district_id.drop(['sum_square_dr', 'sum_roos_dr'], axis=1, inplace=True)

    df_in = pd.merge(df_in, info_by_district_id, on=['DistrictId', 'HouseYear'], how='left')

    df_in['mean_square_per_room_in_dr'] = df_in['mean_square_per_room_in_dr'].fillna(df_in['mean_square_per_room_in_dr'].mean())

    df_in.loc[df_in['Rooms'] > 6, 'Rooms'] = (df_in.loc[df_in['Rooms'] > 6, 'Square'] / df_in.loc[df_in['Rooms'] > 6, 'mean_square_per_room_in_dr']).astype('int')

    df_in.loc[df_in['Rooms'] == 0, 'Rooms'] = (df_in.loc[df_in['Rooms'] == 0, 'Square'] / df_in.loc[df_in['Rooms'] == 0, 'mean_square_per_room_in_dr']).astype('int')

    df_in.loc[df_in['Rooms'] == 0, 'Rooms'] = 1

    return df_in

def square_400_data_farme(df_in):
    df_in.loc[df_in['Square'] > 400, 'Square'] = df_in.loc[df_in['Square'] > 400, 'Square'] / 10
    return df_in

def square_data_frame(df_in):
    info_by_district_id = df_in.groupby(['DistrictId', 'Rooms', 'HouseYear'], as_index=False).agg(
        {'Square': 'mean'}).rename(
        columns={'Square': 'mean_square_rooms_dr'})

    df_in = pd.merge(df_in, info_by_district_id, on=[
        'DistrictId', 'Rooms', 'HouseYear'], how='left')

    df_in.loc[abs(df_in['Square'] - df_in['mean_square_rooms_dr']) > 2 * sigma, 'Square'] \
        = df_in.loc[abs(df_in['Square'] - df_in['mean_square_rooms_dr']) > 2 * sigma, 'Rooms'] \
        * df_in.loc[abs(df_in['Square'] - df_in['mean_square_rooms_dr']) > 2 * sigma, 'mean_square_per_room_in_dr']
    return df_in

def prepare_lifesquare(df_in):
    df_in.loc[df_in['Square'] < df_in['LifeSquare'],
           'LifeSquare'] = df_in.loc[df_in['Square'] < df_in['LifeSquare'], 'Square']
    return df_in

def fillna_life_square(df_in):
    df_in['LifeSquare'] = df_in['LifeSquare'].fillna(df_in['LifeSquare'].mean())
    return df_in

def fix_house_year_manual_data_frame(df_in):
    df_in.loc[df_in['HouseYear'] == 20052011, 'HouseYear'] = int((2005 + 2011) / 2)
    df_in.loc[df_in['HouseYear'] == 4968, 'HouseYear'] = 1968
    return df_in

def del_missing_data_frame(df_in):
    df_na = (df_in.isnull().sum() / len(df_in)) * 100

    df_na = df_na.drop(
        df_na[df_na == 0].index).sort_values(ascending=False)
    df_na = list(df_na.index)
    df_in.drop(df_na, axis=1, inplace=True)
    return df_in

def add_cluster_year(df):
    df_scaled = pd.DataFrame(mm_scaler.fit_transform(
        df.loc[:, ['HouseYear']]), columns=['HouseYear'])
    df['cluster_year'] = cluster_ag_m.fit_predict(df_scaled)
    return df


#

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

print("загрузка данных - выполнено")

print(f"данные train:\t{train.shape[0]}\t {train.shape[1]}")
print(train.head())
print(train.info(memory_usage='deep'))

#оптимизация памяти
train['Rooms'] = train['Rooms'].astype('int64')
train['HouseFloor'] = train['HouseFloor'].astype('int64')

train = opt_data_frame(train)
print(train.info(memory_usage='deep'))


print(f'данные test:\t{test.shape[0]}\t {test.shape[1]}')
print(test.head())
print(test.info(memory_usage='deep'))

test['Rooms'] = test['Rooms'].astype('int64')
test['HouseFloor'] = test['HouseFloor'].astype('int64')

test = opt_data_frame(test)
print(test.info(memory_usage='deep'))

#всего данных
all_data = pd.concat((train, test), sort=False).reset_index(drop=True)
all_data.drop(['Price'], axis=1, inplace=True)
print(all_data.describe().transpose())

#для комнат >5
print(all_data.loc[all_data['Rooms'] > 5])

#для комнат = 0
print(all_data.loc[all_data['Rooms'] == 0])

#площдь > 200
print(all_data.loc[all_data['Square'] > 200].nlargest(20, 'Square'))

# #нормальное распределение
sns.distplot(all_data['Square'], fit=norm)
mu, sigma = norm.fit(all_data['Square'])
print(f'sigma = {sigma:.3f} mu = {mu:.3f}')

# plt.title('распределение')
# plt.ylabel('частота')
#
# fig = plt.figure()
# res = stats.probplot(all_data['Square'], plot=plt)
# plt.show()
#
print(all_data.loc[all_data['HouseYear'] > 2020])
sns.distplot(train['Price'], fit=norm)
mu, sigma = norm.fit(train['Price'])
print(f'mu = {mu:.3f} and sigma = {sigma:.3f}')
#
# plt.title('распределение - цена')
# plt.ylabel('частота')
#
# fig = plt.figure()
# res = stats.probplot(train['Price'], plot=plt)
# plt.show()
#
# #распределение целевого значения
price_log = np.log1p(train['Price'])
sns.distplot(price_log, fit=norm)
mu, sigma = norm.fit(train['Price'])
print(f'mu = {mu:.3f} and sigma = {sigma:.3f}')
#
# plt.title('распределение')
# plt.ylabel('частота')
#
# fig = plt.figure()
# res = stats.probplot(price_log, plot=plt)
# plt.show()

#

all_data = pd.concat((train, test), sort=False).reset_index(drop=True)
all_data.drop(['Price'], axis=1, inplace=True)
print(f'всего данных : {all_data.shape}')

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(
    all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio': all_data_na})

print(missing_data)

#
# correlation = train.loc[:, train.columns != 'Id'].corr(numeric_only = True)
# plt.subplots(figsize=(12, 9))
# sns.heatmap(correlation, vmax=0.9, square=True)
#
# correlation = train.loc[:, train.columns != 'Id'].corrwith(
#     train['Price']).abs().sort_values(ascending=False)[1:]
# plt.bar(correlation.index, correlation.values)
# plt.title('корреляция')
# plt.xticks(rotation='vertical')
# plt.show()

#
mm_scaler = MinMaxScaler()

train_clust = fix_house_year_manual_data_frame(train.copy())
train_cluster_scl = pd.DataFrame(mm_scaler.fit_transform(train_clust.loc[:, ['HouseYear', 'Price']]), columns=['HouseYear', 'Price'])

#
# int = []
# for i in range(2, 10):
#     m_temp = KMeans(n_clusters=i, random_state=100)
#     m_temp.fit(train_cluster_scl)
#     _int = m_temp.inertia_
#     int.append(_int)
#
# plt.plot(range(2, 10), int)
# plt.show()

#
# plt.scatter(train_cluster_scl['HouseYear'], train_cluster_scl['Price'])
# plt.xlabel('HouseYear')
# plt.ylabel('Price')
# plt.show()

#
m_kmeans = KMeans(n_clusters=5, random_state=100)
title_train = m_kmeans.fit_predict(train_cluster_scl)

# plt.scatter(train_cluster_scl['HouseYear'],
#             train_cluster_scl['Price'], c=title_train)
#
# plt.xlabel('HouseYear')
# plt.ylabel('Price')
# plt.show()

cluster_ag_m = AgglomerativeClustering(n_clusters=5)

train_clust['cluster_year'] = cluster_ag_m.fit_predict(
    train_cluster_scl)

# plt.scatter(train_clust['HouseYear'],
#             train_clust['Price'], c=train_clust['cluster_year'])
# plt.xlabel('HouseYear')
# plt.ylabel('Price')
# plt.show()


def add_mean_price(df, df_train=train):
    price = df_train['Price'].mean()
    price_mean_by_rooms = df_train.groupby(['Rooms'], as_index=False).agg({'Price': 'mean'}).\
        rename(columns={'Price': 'mean_price_by_rooms'})

    price_mean_by_distr_rooms = df_train.groupby(['DistrictId', 'Rooms'], as_index=False).agg({'Price': 'mean'}).\
        rename(columns={'Price': 'mean_price_dr'})

    df = pd.merge(df, price_mean_by_distr_rooms, on=[
                  'DistrictId', 'Rooms'], how='left')
    df = pd.merge(df, price_mean_by_rooms, on='Rooms', how='left')
    df['mean_price_dr'] = df['mean_price_dr'].fillna(df['mean_price_by_rooms'])
    df['mean_price_dr'] = df['mean_price_dr'].fillna(price)
    df['mean_price_by_rooms'] = df['mean_price_by_rooms'].fillna(price)
    return df

def add_distr_info(df):
    distr_info = df['DistrictId'].value_counts().reset_index().\
        rename(columns={"index": "DistrictId", "DistrictId": 'large_district'})
    df = pd.merge(df, distr_info, on='DistrictId', how='left')
    df['large_district'] = df['large_district'].fillna(1)
    return df

def data_prepare(df, df_train=train):
    df = square_400_data_farme(df)
    df = fix_house_year_manual_data_frame(df)
    df = fix_room_data_frame(df)
    df = square_data_frame(df)
    df = prepare_lifesquare(df)
    df = fillna_life_square(df)
    df = del_missing_data_frame(df)
    df = add_cluster_year(df)
    df = add_mean_price(df, df_train)
    df = add_distr_info(df)
    df = pd.get_dummies(df)
    df.drop('mean_square_per_room_in_dr', axis=1, inplace=True)
    df.drop('mean_square_rooms_dr', axis=1, inplace=True)
    opt_data_frame(df)
    return df

def data_prepare(df, df_train=train):
    df = square_400_data_farme(df)
    df = fix_house_year_manual_data_frame(df)
    df = fix_room_data_frame(df)
    df = square_data_frame(df)
    df = prepare_lifesquare(df)
    df = fillna_life_square(df)
    df = del_missing_data_frame(df)
    df = add_cluster_year(df)
    df = add_mean_price(df, df_train)
    df = add_distr_info(df)
    df = pd.get_dummies(df)
    df.drop('mean_square_per_room_in_dr', axis=1, inplace=True)
    df.drop('mean_square_rooms_dr', axis=1, inplace=True)
    opt_data_frame(df)
    return df


def model_test(model, name, test, valid):
    model_pred = model.predict(test)
    r2 = r2_score(valid, model_pred)
    mse = mean_squared_error(valid, model_pred)
    plt.scatter(valid, (model_pred - valid))
    plt.xlabel("Predicted values")
    plt.ylabel("Real values")
    plt.title(name)
    plt.legend([f'R2= {r2:.4f} and mse= {mse:.0e}'])
    plt.axhline(0, color='red')
    plt.show()


def model_top_deviation(model, test, valid):
    model_pred = model.predict(test)
    model_test = test.copy()
    model_test['Price'] = model_pred
    model_test['Price_test'] = valid
    model_test['SD'] = abs(model_test['Price']
                           - model_test['Price_test'])
    return model_test.nlargest(10, 'SD')


print(train.columns)
features = list(train.loc[:, train.columns != 'Id'].corrwith(
    train['Price']).abs().sort_values(ascending=False)[1:].index)

target = 'Price'
print(train[features].head())

models_dict = {}

X_train, X_test, y_train, y_test = train_test_split(
    train[features], train[target], test_size=0.3, random_state=42)

X_train = data_prepare(X_train, train)
X_test = data_prepare(X_test, train)

X_train.info()
X_test.info()

print(X_train.head())
print(y_train.head())

#linear
line_regression_model = LinearRegression()
line_regression_model.fit(X_train, y_train)

LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None)

models_dict['Linear Regression'] = line_regression_model
model_test(line_regression_model, 'Linear Regression', X_test, y_test)
model_top_deviation(line_regression_model, X_test, y_test)

#random forest

random_forest_regressor_model = RandomForestRegressor()
random_forest_regressor_model.fit(X_train, y_train)

RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
                      max_features='auto', max_leaf_nodes=None,
                      min_impurity_decrease=0.0,
                      min_samples_leaf=1, min_samples_split=2,
                      min_weight_fraction_leaf=0.0, n_estimators=10,
                      n_jobs=None, oob_score=False, random_state=None,
                      verbose=0, warm_start=False)

models_dict['Random Forest Regressor'] = random_forest_regressor_model
model_test(random_forest_regressor_model, 'Random Forest Regressor', X_test, y_test)
model_top_deviation(random_forest_regressor_model, X_test, y_test)

#GradientBoostingRegressor
gradient_boosting_regressor_model = GradientBoostingRegressor()
gradient_boosting_regressor_model.fit(X_train, y_train)
GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                          learning_rate=0.1, loss='ls', max_depth=3,
                          max_features=None, max_leaf_nodes=None,
                          min_impurity_decrease=0.0,
                          min_samples_leaf=1, min_samples_split=2,
                          min_weight_fraction_leaf=0.0, n_estimators=100,
                          n_iter_no_change=None,
                          random_state=None, subsample=1.0, tol=0.0001,
                          validation_fraction=0.1, verbose=0, warm_start=False)
models_dict['Gradient Boosting Regressor'] = gradient_boosting_regressor_model
model_test(gradient_boosting_regressor_model,'Gradient Boosting Regressor', X_test, y_test)
model_top_deviation(gradient_boosting_regressor_model, X_test, y_test)

#LassoCV
lasso_cv_model = LassoCV()
lasso_cv_model.fit(X_train, y_train)

LassoCV(alphas=None, copy_X=True, cv='warn', eps=0.001, fit_intercept=True,
        max_iter=1000, n_alphas=100, n_jobs=None,
        positive=False, precompute='auto', random_state=None,
        selection='cyclic', tol=0.0001, verbose=False)
models_dict['LassoCV'] = lasso_cv_model
model_test(lasso_cv_model, 'LassoCV', X_test, y_test)
model_top_deviation(lasso_cv_model, X_test, y_test)

#LGBMRegressor
lgbm_regressor_model = LGBMRegressor()
lgbm_regressor_model.fit(X_train, y_train)
LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
              importance_type='split', learning_rate=0.1, max_depth=-1,
              min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
              n_estimators=100, n_jobs=-1, num_leaves=31, objective=None,
              random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True,
              subsample=1.0, subsample_for_bin=200000, subsample_freq=0)
model_test(lgbm_regressor_model, 'LGBMRegressor', X_test, y_test)
model_top_deviation(lgbm_regressor_model, X_test, y_test)
model_test(lgbm_regressor_model, 'LGBMRegressor', X_test, y_test)

#XGBRegressor
xgboost_model = XGBRegressor()
xgboost_model.fit(X_train, y_train)
XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0,
             importance_type='gain', learning_rate=0.1, max_delta_step=0,
             max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
             n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=None, subsample=1, verbosity=1)
models_dict['XGBRegressor'] = xgboost_model
model_test(xgboost_model, 'XGBRegressor', X_test, y_test)
model_top_deviation(xgboost_model, X_test, y_test)

#

def models_r2(models, test, valid):
    scores = pd.DataFrame(columns=['name', 'r2', 'mse'])
    for name, model in models.items():
        test_pred = model.predict(test)
        r2 = r2_score(valid, test_pred)
        mse = mean_squared_error(valid, test_pred)
        scores = scores.append(
            {'name': name, 'r2': r2, 'mse': mse}, ignore_index=True)
    scores.sort_values('r2', ascending=False, inplace=True)
    return scores

models_score_test = models_r2(models_dict, X_test, y_test)
models_score_train = models_r2(models_dict, X_train, y_train)

print(models_score_test[['name', 'r2']])

r2_max_test = models_score_test['r2'].max()
r2_max_train = models_score_train['r2'].max()
plt.barh(models_score_test['name'], models_score_test['r2'],
         alpha=0.5, color='red', label=f'Test  Data: R2 max: {r2_max_test:.4f}')
plt.barh(models_score_train['name'], models_score_train['r2'],
         alpha=0.5, color='grey', label=f'Train Data: R2 max: {r2_max_train:.4f}')
plt.title('R2')
plt.legend()
plt.axvline(0.6, color='red')
plt.axvline(r2_max_test, color='yellow')
plt.show()

mse_min_test = models_score_test['mse'].min()
mse_min_train = models_score_train['mse'].min()
plt.barh(models_score_test['name'], models_score_test['mse'],
         alpha=0.5, color='red', label=f'Test  Data MSE min: {mse_min_test:.0e}')
plt.barh(models_score_train['name'], models_score_train['mse'],
         alpha=0.5, color='grey', label=f'Train Data MSE min: {mse_min_train:.0e}')
plt.title('Mean squared error')
plt.legend(loc=2)
plt.axvline(mse_min_test, color='yellow')
plt.show()

best_model = models_dict['Random Forest Regressor']
pd.DataFrame({'name': list(X_train.columns),
              'importances': list(best_model.feature_importances_)})

model_test(best_model, 'Random Forest Regressor', X_test, y_test)

#
test = data_prepare(test)
test_features = list(X_train.columns)
test[test_features].info()

test['Price'] = best_model.predict(test[test_features])
price_log = np.log1p(test['Price'])
sns.distplot(price_log, fit=norm)

mu, sigma = norm.fit(test['Price'])

print(f'mu = {mu:.2f} and sigma = {sigma:.2f}')

plt.legend(
    [f'Normal dist. ($\mu=$ {mu:.2f} and $\sigma=$ {sigma:.2f} )'], loc='best')
plt.ylabel('Frequency')
plt.title('Price distribution')

fig = plt.figure()
res = stats.probplot(price_log, plot=plt)
plt.show()

test[['Id', 'Price']].to_csv('bondarenko_predictions.csv', index=None)

