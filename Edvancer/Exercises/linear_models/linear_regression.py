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
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin

from operator import itemgetter

class Linear_Regression:
    def __init__(self, file):
        self.df = pd.read_csv(file)
        self.cyclic_cols = ['Post Published Weekday','Base Date Time Weekday']
        cat_columns = (self.df.select_dtypes('O')).columns
        self.cat_columns = np.setdiff1d(cat_columns, self.cyclic_cols)

    def del_df_cols(self, cols):
        self.df.drop(cols, axis=1, inplace=True)

    def set_X_train(self,param):
        self.X_train = param
    
    def set_X_test(self,param):
        self.X_test = param
    
    def set_y_train(self,param):
        self.y_train = param
    
    def set_y_test(self,param):
        self.y_test = param
    
    def impute_missing(self):
        impute_missing_cols = []

        for cols in self.df.columns:
            if self.df.loc[:,cols].isnull().sum() > 0:
                impute_missing_cols.append(cols)
                if self.df.loc[:,cols].dtype == 'O':
                    self.df[cols] = self.df[cols].modal()
                else:
                    self.df[cols] = self.df[cols].fillna(self.df[cols].mean())
        if len(impute_missing_cols) == 0:
            print("columns have no missing values")
        else:
            print('#############################')
            print(impute_missing_cols)

        return self.df

    def convert_toInt(self):
        convert_to_int = []
        df = self.df.copy()
        df.drop(self.cat_columns, axis=1, inplace=True)
        for col in df:
            if df[col].dtype == 'float64' or df[col].dtype == 'uint8':
                convert_to_int.append(col)
                self.df[col] = df[col].astype(float).astype('int64')
        print(convert_to_int)
        return self.df   

    def get_dummyData(self, cutoff):
        dummify = categorical_dummies(cutoff, file)
        dummify.fit()
        self.df = dummify.transform()

    def days_toNumbers(self):
        days_list = ['Post Published Weekday','Base Date Time Weekday']
        days = {'Monday':1,'Tuesday':2, 'Wednesday':3, 'Thursday':4, 'Friday':5, 'Saturday':6, 'Sunday':7}

        for days_col in days_list:
            self.df[days_col] = self.df[days_col].map(days)
            print(self.df.shape)
        return self.df

    def cyclic_features(self):
        cyclic_list = self.cyclic_cols
        for cyclic_col in cyclic_list:
            self.df[cyclic_col+'_sin']=np.sin(2*np.pi*self.df[cyclic_col])/7
            self.df[cyclic_col+'_cos']=np.cos(2*np.pi*self.df[cyclic_col])/7

            self.df = self.df.drop(cyclic_col, axis=1)
        print(self.df.shape)
        return self.df

    def remove_outliers(self,X_train, percentile_value, outlier_columns):
        for col in outlier_columns:
            X_train[col] = np.where(X_train[col] > percentile_value, X_train[col].mean(), X_train[col])
            self.X_train = X_train

        return X_train

    def best_feature_no(self,model_lr, scoring):
        # model_lr.fit(X_train, y_train)

        # Use grid searchCV to find best number of features
        param_grid = {'n_features_to_select': np.arange(0,85,5)}
        folds = KFold(n_splits=5, shuffle=True, random_state=100)

        feature_search = GridSearchCV(estimator = RFE(model_lr), param_grid = param_grid, scoring = scoring , cv = folds, return_train_score=True)
        feature_search.fit(self.X_train, self.y_train)
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

    def RFE_model(self,estimator, feature_number, scoring):
        # print(X_train[X_train.isnull().sum().index])
        model_lr = estimator

        model_lr.fit(self.X_train, self.y_train)
        # TODO: try ridge, lasso and eccentric and see the best performaer
        # TODO: play around with the plots
        # feature selection
        feature_number = feature_number #best_feature_no(model_lr)
        rfe_object = RFE(model_lr, n_features_to_select=feature_number)
        rfe_object.fit(self.X_train, self.y_train)

        print(rfe_object.ranking_)
        column = self.X_train.columns.to_list()
        for x,y in sorted(zip(rfe_object.ranking_, column), key = itemgetter(0)):
            # show 1st 10 features
            print(x,y)
            if x == feature_number + 3:
                break

        # Predict 
        y_train_pred = rfe_object.predict(self.X_train)
        y_test_pred = rfe_object.predict(self.X_test)

        # # check line of best fit
        # plt.figure()
        # plt.scatter(self.X_train.loc[:,'p3__feat7'], self.y_train)
        # plt.plot(self.X_train.loc[:,'p3__feat7'], y_train_pred)
        # plt.show()

        # Performance of the selected 10 features
        if scoring == 'r2':
            train_score1 = r2_score(self.y_train, y_train_pred)
            test_score1 = r2_score(self.y_test, y_test_pred)

            # print('best estimator = ', rfe_object.best_estimator_)
        elif scoring == 'neg_mean_absolute_error':
            train_score1 = mean_absolute_error(self.y_train, y_train_pred)
            test_score1 = mean_absolute_error(self.y_test, y_test_pred)
        
        print (f'training data prediction score for {scoring} is = {train_score1}') # = 0.3202592463942481
        print (f'test data prediction score for {scoring} is = {test_score1}') # = 0.2975885376233065
        mse = mean_squared_error(self.y_train, y_train_pred) #857.2621770736797
        print(f'accuracy of model = {mse}')

    def view_features_outliers(self,data):
        sns.boxplot(data)

class categorical_dummies(BaseEstimator, TransformerMixin, Linear_Regression):
    def __init__(self, freq_cutoff, file):
        super().__init__(file)
        self.frequency_cutoff = freq_cutoff
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


# file = r'D:\AI_ML\AI\Machine Learning in Python\data\data\facebook_comments.csv'
file = r'/mnt/d/AI_ML/AI/Machine Learning in Python/data/data/facebook_comments.csv'

lr_model = Linear_Regression(file)

# clean Data 
lr_model.impute_missing()
lr_model.convert_toInt()
lr_model.get_dummyData(500)
lr_model.days_toNumbers()
lr_model.cyclic_features()

# Split data
y = lr_model.df['Comments_in_next_H_hrs']
X = lr_model.df.drop('Comments_in_next_H_hrs', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7)

# Set the data into the class
lr_model.set_X_train(X_train)
lr_model.set_X_test(X_test)
lr_model.set_y_train(y_train)
lr_model.set_y_test(y_test)

# Scaling the data
feature_scaler = StandardScaler()
X_train_list = feature_scaler.fit_transform(X_train.values)  #this return a numpy.ndarray that we change back to dataframe in next line
X_train = pd.DataFrame(X_train_list, index=X_train.index, columns=X_train.columns)
lr_model.set_X_train(X_train)

X_test_list = feature_scaler.transform(X_test.values)
X_test = pd.DataFrame(X_test_list, index=X_test.index, columns=X_test.columns)
lr_model.set_X_test(X_test)

# Removing outliers from X_train
outlier_columns = ['likes', 'talking_about', 'feat5', 'feat10', 'feat12', 'feat13', 'feat15', 'feat18', 'feat20', 'feat22','feat27', 'feat28', 'Post_Length', 'Post Share Count']

percentile_value = X_train.iloc[:,0].quantile(0.99)
X_train = lr_model.remove_outliers(X_train, percentile_value, outlier_columns)
lr_model.set_X_train(X_train)

plt.figure()
plt.scatter(X_train.iloc[:,0], y_train)
plt.show()

# Types of models and scoring to test
model_lr = LinearRegression()   #train_r2_score= 0.3202592463942481,  test_r2_score= 0.2975885376233065
model_ridge = Ridge()
model_lasso = Lasso()   

scoring_r2 = 'r2'
scoring_MAE = 'neg_mean_absolute_error'

# Cross validation
scores = cross_val_score(model_lasso, X_train, y_train, scoring=scoring_r2, cv=5) # = 0.3064387581535156
print(scores) 
print(f'%%%%%%%%%%%%%%%%%%%%%% Expected score = {scores.mean()}') # Gives a rough idea of what the score will be

# Best_feature selection
# feature_number = lr_model.best_feature_no(model_lasso, scoring_MAE)
feature_no = 55 # feature_number['n_features_to_select']

# Linear model with RFE
lr_model.RFE_model(model_lasso, feature_no, scoring_r2)
# %%
