from pprint import pprint
from tabnanny import verbose
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score

from operator import itemgetter

file = r'/mnt/d/AI_ML/AI/Machine Learning in Python/data/data/facebook_comments.csv'

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


# print(X_train[X_train.isnull().sum().index])
model_lr = LinearRegression()

# TODO: try ridge, lasso and eccentric and see the best performaer
# TODO: play around with the maps
# feature selection
feature_number = 45
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
train_score1 = r2_score(y_train, y_train_pred)
test_score1 = r2_score(y_test, y_test_pred)

print ('training data prediction score = ',train_score1)
print ('test data prediction score = ',test_score1)

# Use grid searchCV to find best number of features
param_grid = {'n_features_to_select':[5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]}
feature_search = GridSearchCV(estimator = RFE(model_lr), param_grid = param_grid, scoring = 'r2' , cv = 5)
# feature_search.fit(X_train, y_train)

# print(feature_search.best_params_)
# feature_number = 45
# model_lr_featured = GridSearchCV(estimator=model_lr, param_grid={'feature_names':[5,10,20,30,40,50,60,70,80]})
# model_lr_featured.fit(X_train, y_train)
# model_lr.fit(X_train, y_train)
# print(model_lr.intercept_)

# print(list(zip(x_train.columns, model_lr.coef_)))


# converted_to_int = convert_toInt()
# print(converted_to_int)