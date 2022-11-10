import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
# import '../../pipes.py'
from pipes import *
# from ...pipes import * 

# file = r'D:\AI_ML\AI\Machine Learning in Python\data\data\paydayloan_collections.csv'
file = r'/mnt/d/AI_ML/AI/Machine Learning in Python/data/data/paydayloan_collections.csv'
df = pd.read_csv(file)

target_col = 'payment'
target_df = df[target_col]
numeric_cols = df.select_dtypes(exclude='O')
categorical_cols = list(df.loc[:,df.columns != 'payment'].select_dtypes('O').columns)


p1=pdPipeline([
    ('var_select', VarSelector(list(numeric_cols.columns))),
    ('convert_to_numeric', convert_to_numeric()),
    ('impute_missing', DataFrameImputer())
])

p2=pdPipeline([
    ('var_select', VarSelector(categorical_cols)),
    ('impute_missing', DataFrameImputer()),
    ('get_dummies', get_dummies_Pipe())
])

data_pipe = FeatureUnion([
    ('to_numeric', p1),
    ('dummify',p2)
])

# data_pipe.fit(df)
df.loc[:,target_col] = np.where(df[target_col] == 'Success', 1, 0)
y = pd.DataFrame(df.iloc[:,df.columns.get_loc(target_col)])
# print(y)
# print(pd.DataFrame(y))

df.drop(target_col,axis=1, inplace=True)
X = pd.DataFrame(df)
print('-------------x--------- ',y.value_counts())
# .drop('payment', axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


clf = Pipeline(
    steps=[("preprocessor", data_pipe), ("classifier", RandomForestClassifier())]
)

clf.fit(X_train, y_train)
print("model score: %.3f" % clf.score(X_test, y_test)) # = 0.858

# print(p1)