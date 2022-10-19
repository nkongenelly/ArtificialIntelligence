#%%
from tabnanny import verbose
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn. linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from operator import itemgetter

file = r'D:\AI_ML\AI\Machine Learning in Python\data\data\facebook_comments.csv'
# file = r'/mnt/d/AI_ML/AI/Machine Learning in Python/data/data/facebook_comments.csv'

facebook_comments = pd.read_csv(file)

# print(facebook_comments.head())

def impute_missing(facebook_comments):
    # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    # imp = SimpleImputer(strategy="most_frequent")
    # imp.fit(facebook_comments)

    impute_missing_cols = []

    for cols in facebook_comments.columns:
        if facebook_comments.loc[:,cols].isnull().sum() > 0:
            impute_missing_cols.append(cols)
            if facebook_comments.loc[:,cols].dtype == 'O':
                facebook_comments[cols] = facebook_comments[cols].modal()
            else:
                facebook_comments[cols] = facebook_comments[cols].fillna(facebook_comments[cols].mean())
    if len(impute_missing_cols) == 0:
        print("columns have no missing values")
    else:
        print('#############################')
        print(impute_missing_cols)

    return facebook_comments

def convert_toInt(facebook_comments):
    convert_to_int = []
    for col in facebook_comments:
        if facebook_comments[col].dtype == 'float64' or facebook_comments[col].dtype == 'uint8':
            convert_to_int.append(col)
            facebook_comments[col] = facebook_comments[col].astype(float).astype('int64')
    print(convert_to_int)
    return facebook_comments   

def get_dummyData(facebook_comments, cutoff):
    categories = facebook_comments['page_category'].value_counts()

    page_category_mode = facebook_comments['page_category'].mode()[0]
    facebook_comments['page_category'] = np.where(categories[facebook_comments['page_category']] <= cutoff, page_category_mode , facebook_comments['page_category'])

    categorical = pd.get_dummies(facebook_comments['page_category'], prefix='page_category', drop_first=True)
    # categorical = convert_toInt(categorical)

    facebook_comments = pd.concat([facebook_comments.reset_index(drop=True),categorical.reset_index(drop=True)], axis = 1)
    facebook_comments = facebook_comments.drop('page_category', axis=1)

    return facebook_comments

def days_toNumbers(facebook_comments):
    days_list = ['Post Published Weekday','Base Date Time Weekday']
    days = {'Monday':1,'Tuesday':2, 'Wednesday':3, 'Thursday':4, 'Friday':5, 'Saturday':6, 'Sunday':7}

    for days_col in days_list:
        facebook_comments[days_col] = facebook_comments[days_col].map(days)
        print(facebook_comments.shape)
    
    return facebook_comments

def cyclic_features(facebook_comments):
    cyclic_list = ['Post Published Weekday','Base Date Time Weekday']
    for cyclic_col in cyclic_list:
        facebook_comments[cyclic_col+'_sin']=np.sin(2*np.pi*facebook_comments[cyclic_col])/7
        facebook_comments[cyclic_col+'_cos']=np.cos(2*np.pi*facebook_comments[cyclic_col])/7

        facebook_comments = facebook_comments.drop(cyclic_col, axis=1)
        # del facebook_comments[cyclic_col]
    print(facebook_comments.shape)
    return facebook_comments

def best_feature_no(model_lr, scoring):
    model_lr.fit(X_train, y_train)

    # Use grid searchCV to find best number of features
    param_grid = {'n_features_to_select': np.arange(0,85,5)}
    folds = KFold(n_splits=5, shuffle=True, random_state=100)

    feature_search = GridSearchCV(estimator = RFE(model_lr), param_grid = param_grid, scoring = scoring , cv = folds, return_train_score=True)
    feature_search.fit(X_train, y_train)
    feature_number = feature_search.best_params_
    print( feature_number)
    print('------------cv_results = ---------')
    cv_results = pd.DataFrame(feature_search.cv_results_)

    print(cv_results.columns)

    plt.figure(figsize=(16,6))

    plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_test_score"])
    plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_train_score"])
    plt.xlabel('number of features')
    plt.ylabel('r-squared')
    plt.title("Optimal Number of Features")
    plt.legend(['test score', 'train score'], loc='upper left')
    plt.show()

    return feature_number

def RFE_model(estimator, feature_number, scoring):
    # print(X_train[X_train.isnull().sum().index])
    model_lr = estimator
    model_lr.fit(X_train, y_train)
    # TODO: try ridge, lasso and eccentric and see the best performaer
    # TODO: play around with the maps
    # feature selection
    feature_number = feature_number #best_feature_no(model_lr)
    rfe_object = RFE(model_lr, n_features_to_select=feature_number)
    rfe_object.fit(X_train, y_train)

    print(rfe_object.ranking_)
    column = X_train.columns.to_list()
    for x,y in sorted(zip(rfe_object.ranking_, column), key = itemgetter(0)):
        # show 1st 10 features
        print(x,y)
        if x == feature_number + 3:
            break

    # Predict 
    y_train_pred = rfe_object.predict(X_train)
    y_test_pred = rfe_object.predict(X_test)

    # Performance of the selected 10 features
    if scoring == 'r2':
        train_score1 = r2_score(y_train, y_train_pred)
        test_score1 = r2_score(y_test, y_test_pred)

        # print('best estimator = ', rfe_object.best_estimator_)
    elif scoring == 'neg_mean_absolute_error':
        train_score1 = mean_absolute_error(y_train, y_train_pred)
        test_score1 = mean_absolute_error(y_test, y_test_pred)
    
    print (f'training data prediction score for {scoring} is = {train_score1}') # = 0.3202592463942481
    print (f'test data prediction score for {scoring} is = {test_score1}') # = 0.2975885376233065



facebook_comments = impute_missing(facebook_comments)
facebook_comments = convert_toInt(facebook_comments)
facebook_comments = get_dummyData(facebook_comments, 200)
facebook_comments = days_toNumbers(facebook_comments)
facebook_comments = cyclic_features(facebook_comments)

y = facebook_comments['Comments_in_next_H_hrs']
X = facebook_comments.drop('Comments_in_next_H_hrs', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.3)

# Scaling the data
feature_scaler = StandardScaler()
X_train_list = feature_scaler.fit_transform(X_train.values)  #this return a numpy.ndarray that we change back to dataframe in next line
X_train = pd.DataFrame(X_train_list, index=X_train.index, columns=X_train.columns)

X_test_list = feature_scaler.transform(X_test.values)
X_test = pd.DataFrame(X_test_list, index=X_test.index, columns=X_test.columns)


model_lr = LinearRegression()   #train_r2_score= 0.3202592463942481,  test_r2_score= 0.2975885376233065
model_ridge = Ridge()
#train_r2_score= 0.3564303265952593,  test_r2_score= 0.24465573273899666
#train_MAE_score= 7.382574073988469,  test_MAE_score= 6.734786603682472

model_lasso = Lasso()   
#train_r2_score= 0.33684651651352526,  test_r2_score= 0.2819304519846273
#train_MAE_score= 6.717708472548399,  test_MAE_score= 7.378769880326644

scoring_r2 = 'r2'
scoring_MAE = 'neg_mean_absolute_error'

# Cross validation
scores = cross_val_score(model_lasso, X_train, y_train, scoring=scoring_MAE, cv=5) # = 0.3064387581535156
print(scores) 
print(f'%%%%%%%%%%%%%%%%%%%%%% Expected score = {scores.mean()}') # Gives a rough idea of what the score will be

feature_number = best_feature_no(model_lasso, scoring_MAE)
RFE_model(model_lasso, feature_number['n_features_to_select'], scoring_MAE)

# lasso() gives best results

# %%
