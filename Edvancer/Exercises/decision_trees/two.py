import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

class Random_Forest:
    def __init__(self, file):
        self.file = file
        self.df = pd.read_csv(file)
        self.target_col = 'payment'
        self.numeric_cols = self.df.loc[:,self.df.columns != 'payment'].select_dtypes(exclude='O')
        categorical_cols = list(self.df.select_dtypes('O').columns)

    def set_X_train(self,param):
        self.X_train = param
    
    def set_X_test(self,param):
        self.X_test = param
    
    def set_y_train(self,param):
        self.y_train = param
    
    def set_y_test(self,param):
        self.y_test = param

    def target_to_binary(self):
        self.target_column = 'payment'
        self.target_func = lambda :np.where(self.df[self.target_column] == 'Success', 1, 0)
        self.target_transformer = Pipeline(steps = 
                [
                    ('impute_missing', SimpleImputer(strategy="mode")),
                    ('convert_target_to_binary', self.target_func())
                ])
    
    def convert_to_numeric(self):
        for col in self.numeric_cols:
            self.df[col] = self.df[col].astype('int')

    def target_numeric(self):
        self.numeric_func = self.convert_to_numeric()
        self.numeric_transformers = Pipeline(steps = 
                [
                    ('impute_missing', SimpleImputer(strategy="mean")),
                    ('numeric', self.numeric_func),
                    ('scaler', StandardScaler())
                ])

    def categorical_data(self):
        self.categorical_cols = list(self.df.select_dtypes('O').columns)
        self.categorical_func = categorical_dummies(0, self.file)
        self.categorical_transformers = Pipeline(steps = 
                [
                    ('impute_missing', SimpleImputer(strategy="mean")),
                    ('encoding', self.categorical_func)
                ])

    def preprocessor(self):
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("target_binary", self.target_transformer, self.target_column),
                ("numeric", self.numeric_transformers, self.numeric_cols),
                ("category", self.categorical_transformers, self.categorical_cols)
            ]
        )
        return self.preprocessor

    def model_pipeline(self, classifier):
        self.clf = Pipeline(
            steps=[("preprocessor", self.preprocessor), ("classifier", classifier)]
        )

        self.clf.fit(self.X_train, self.y_train)
        print("model score: %.3f" % self.clf.score(self.X_test, self.y_test))

class categorical_dummies(BaseEstimator, TransformerMixin, Random_Forest):
    def __init__(self, frequency_cutoff, file):
        super().__init__(file)
        self.frequency_cutoff = frequency_cutoff
        self.var_cat_dict = {}
        self.feature_names = []

    def fit(self, X=None, y=None):
        X = self.df
        cat_columns = self.cat_columns
        columns = X[cat_columns].columns
        for col in columns:
            counts = X[col].value_counts()

            if (counts.values < self.frequency_cutoff).sum() == 0:
                cats = counts.index[:-1]
            else:
                cats = counts.index[counts.values > self.frequency_cutoff]
        
            self.var_cat_dict[col] = cats

        for col in self.var_cat_dict.keys():
            for cat in self.var_cat_dict[col]:
                name = col + "_" + str(cat)
                self.feature_names.append(name)

    def transform(self, X=None, y=None):
        X = self.df
        dummy_data = X.copy()

        for col in self.var_cat_dict.keys():
            for cat in self.var_cat_dict[col]:
                name = col + "_" + str(cat)
                dummy_data[name] = (dummy_data[col]==cat).astype(int)

            del dummy_data[col]
            X.drop(col, axis=1, inplace=True)

        return dummy_data


# file = r'D:\AI_ML\AI\Machine Learning in Python\data\data\paydayloan_collections.csv'
file = r'/mnt/d/AI_ML/AI/Machine Learning in Python/data/data/paydayloan_collections.csv'
r_model = Random_Forest(file)
print('df hape before = ', r_model.df.shape)

# clean the data 
r_model.target_to_binary()
r_model.target_numeric()
r_model.categorical_data()
a = r_model.preprocessor()
print(a)

# # split the data
X = r_model.df.drop('payment', axis=1)
y = r_model.df['payment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Set the data into the class
r_model.set_X_train(X_train)
r_model.set_X_test(X_test)
r_model.set_y_train(y_train)
r_model.set_y_test(y_test)

# build and train the model
classifier = RandomForestClassifier()
b = r_model.model_pipeline(classifier)

print('df shape after = ', r_model.df.shape)
