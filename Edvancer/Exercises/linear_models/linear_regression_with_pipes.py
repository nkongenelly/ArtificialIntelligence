import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import fbeta_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline, FeatureUnion
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import NotFittedError

from pipes import *
import time
import pickle

# file = r'D:\AI_ML\AI\Machine Learning in Python\data\data\facebook_comments.csv'
file = r'/mnt/d/AI_ML/AI/Machine Learning in Python/data/data/facebook_comments.csv'
df = pd.read_csv(file)

target_col = ['Comments_in_next_H_hrs']
target_df = df[target_col[0]]
df.drop(target_col[0], axis=1, inplace=True)
numeric_cols = list(df.select_dtypes(exclude='O').columns)
cyclic_cols = ['Post Published Weekday','Base Date Time Weekday']


days = {'Monday':1,'Tuesday':2, 'Wednesday':3, 'Thursday':4, 'Friday':5, 'Saturday':6, 'Sunday':7}

for days_col in cyclic_cols:
    df[days_col] = df[days_col].map(days)
    # print(df[days_col].dtype)

cat_cols_all = df.drop(cyclic_cols, axis=1)
cat_cols = list(cat_cols_all.select_dtypes('O').columns)

p1 = pdPipeline([
    ('var_select', VarSelector(numeric_cols)),
    ('numeric_converter', convert_to_numeric()),
    ('impute_missing', DataFrameImputer())
])

p2 = pdPipeline([
    ('var_select', VarSelector(cat_cols)),
    ('numeric_converter', get_dummies_Pipe()),
    ('impute_missing', DataFrameImputer())
])

p3 = pdPipeline([
    ('var_select', VarSelector(cyclic_cols)),
    ('numeric_converter', cyclic_features_custom()),
    ('impute_missing', DataFrameImputer())
])

data_pipe = FeatureUnion([
    ('to_numeric',p1),
    ('categorical', p2),
    ('cyclic', p3)
])

data_pipe.fit(df)




X = pd.DataFrame(data=data_pipe.transform(df), columns = data_pipe.get_feature_names())
# X = df
y = target_df

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# # lr_model = Pipeline(
# #     steps=[("preprocessor", data_pipe), ("classifier", LinearRegression())]
# # )

# # lr_model.fit(X_train, y_train)

# # y_train_pred = lr_model.predict(X_train)
# # y_test_pred = lr_model.predict(X_test)
# # # if scoring == 'r2':
# # train_score1 = r2_score(y_train, y_train_pred)
# # test_score1 = r2_score(y_test, y_test_pred)

# # print(f'r2_score training = {train_score1}')    #0.342
# # print(f'r2_score testing = {test_score1}')

# # print('best estimator = ', rfe_object.best_estimator_)
# # elif scoring == 'neg_mean_absolute_error':
# # train_score1 = mean_absolute_error(self.y_train, y_train_pred)
# # test_score1 = mean_absolute_error(self.y_test, y_test_pred)

# pipelines = {
#     'rf': Pipeline([('rf',RandomForestRegressor(random_state=1234))]),
#     'gb': Pipeline([('gb',GradientBoostingRegressor(random_state=1234))]),
#     'lasso': Pipeline([('lasso',Lasso(random_state=1234))]),
#     'ridge': Pipeline([('ridge',Ridge(random_state=1234))]),
#     'lr': Pipeline([('lr',LinearRegression())]),
#     'enet': Pipeline([('enet',ElasticNet(random_state=1234))])
# }

# hypergrid = {
#     'rf': {
#         'min_samples_split': [2,4,6],
#         'min_samples_leaf': [1,2,3]
#     },
#     'gb': {
#         'min_samples_split': [2,4,6],
#         'min_samples_leaf': [1,2,3]
#     },
#     'lasso': {
#         'alpha':np.linspace(0,1,10)
#     },
#     'ridge': {
#         'alpha':np.linspace(0,1,10)
#     },
#     'lr': {
#         'n_jobs':[None]
#     },
#     'enet': {
#         'alpha':np.linspace(0,1,10)
#     }
# }

# fit_models = {}

# for algo, pipeline in pipelines.items():
#     model = GridSearchCV(pipeline[0], hypergrid[algo], cv=5, n_jobs=-1)
#     try:
#         print(f'Starting training for {algo}')
#         model.fit(X_train, y_train)
#         fit_models[algo] = model
#         print(f'{algo} has been successfully fit')
#     except NotFittedError as e:
#         print(repr(e))

# # evaluation
# for algo, model in fit_models.items():
#     yhat = model.predict(X_test)
#     print(f'{algo} scores: r2 = {r2_score(y_test,yhat)}, ....MAE = {mean_absolute_error(y_test, yhat)}')
# # rf scores: r2 = 0.6389216759235878, ....MAE = 3.9361882083494444
# # gb scores: r2 = 0.6554272491881054, ....MAE = 4.287141043800574
# # lasso scores: r2 = 0.2965239428373827, ....MAE = 8.162620386609314
# # ridge scores: r2 = 0.295280734534748, ....MAE = 8.33370266406075
# # lr scores: r2 = 0.2953000967022581, ....MAE = 8.334319210893911
# # enet scores: r2 = 0.2971741245790557, ....MAE = 8.201887493819882

# # Save model
# with open('linear_pipeline_model.pkl', 'wb') as f:
#     pickle.dump(fit_models, f)

# read saved pipeline model
with open('linear_pipeline_model.pkl', 'rb') as f:
    fit_models_reloaded = pickle.load( f) 

for algo, model in fit_models_reloaded.items():
    yhat = model.predict(X_test)
    print(f'{algo} scores: r2 = {r2_score(y_test,yhat)}, ....MAE = {mean_absolute_error(y_test, yhat)}') 

# rf scores: r2 = 0.7988674713702559, ....MAE = 2.703811110586599
# gb scores: r2 = 0.7161194728398741, ....MAE = 4.103472386988127
# lasso scores: r2 = 0.31479017448134494, ....MAE = 8.345511555468995
# ridge scores: r2 = 0.3219055920242999, ....MAE = 8.469012826540125
# lr scores: r2 = 0.321934754444537, ....MAE = 8.4694049819397
# enet scores: r2 = 0.3155501543844732, ....MAE = 8.380295752973801