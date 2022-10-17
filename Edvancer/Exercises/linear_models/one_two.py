import pandas as pd
import numpy as np

file = r'/mnt/d/AI_ML/AI/Machine Learning in Python/data/data/facebook_comments.csv'

facebook_comments = pd.read_csv(file)

# print(facebook_comments.head())

def impute_missing(facebook_comments):
    impute_missing_cols = []

    for cols in facebook_comments.columns:
        if facebook_comments.loc[:,cols].isnull().sum() > 0:
            impute_missing_cols.append(cols)

    if len(impute_missing_cols) == 0:
        print("columns have no missing values")
    return facebook_comments

def convert_toInt(facebook_comments):
    convert_to_int = []
    for col in facebook_comments:
        if facebook_comments[col].dtype == 'float64':
            convert_to_int.append(col)
            facebook_comments[col] = facebook_comments[col].astype(float).astype('int64')
    print(convert_to_int)
    return facebook_comments   

def get_dummyData(facebook_comments, cutoff):
    counts = pd.value_counts(facebook_comments['page_category'])
    mask = facebook_comments['page_category'].isin(counts[counts > cutoff].index)
    
    categorical = pd.get_dummies(data = facebook_comments['page_category'][mask], columns=['page_category'], drop_first=True)
    facebook_comments = pd.concat([facebook_comments,categorical], axis = 1)
    facebook_comments = facebook_comments.drop('page_category', axis=1)
    print(facebook_comments.shape)
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


# converted_to_int = convert_toInt()
# print(converted_to_int)