import pandas as pd
import numpy as np
import matplotlib
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class Var_Selector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        return x[self.feature_names]
    def get_feature_names(self):
        return self.feature_names
class Convert_Datetime_Type(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names = []
    def fit(self,x,y=None):
        self.feature_names = x.columns
        return self
    def transform(self, x):
        for col in x.columns:
            x[col] = pd.to_datetime(x[col], errors='coerce')
        return x
    def get_feature_names(self):
        return self.feature_names

class Convert_To_numeric(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names = []
    def fit(self,x,y=None):
        self.feature_names = x.columns
        return self
    def transform(self,x):
        for col in x.columns:
            x[col]= pd.to_numeric(x[col],errors='coerce')
        return x
    def get_feature_names(self):
        return self.feature_names

class ExtractDatetimeCyclic(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names = []
        self.frequencies = {'weekday':7,'month':12,'day_pf_month':31}
        self.week_freq=7
        self.month_freq=12
        self.month_day_freq=31
    def fit(self,x,y=None):
        for col in x.columns:
            for kind in ['weekday','month','day_of_month']:
                self.feature_names.extend([col+'_'+kind+'_'+temp for temp in ['_cos','_sin']])
        return self
    def transform(self,x):
        for col in x.columns:
            weekday = x[col].dt.weekday #Monday = 0 and Sunday = 6
            month = x[col].dt.month #January = 1 and December = 12
            day_of_month = x[col].dt.day

            x[col+'_'+'week'+'_sin']=np.sin(2*np.pi*weekday)/self.week_freq
            x[col+'_'+'week'+'_cos']=np.cos(2*np.pi*weekday)/self.week_freq

            x[col+'_'+'month'+'_sin']=np.sin(2*np.pi*month)/self.month_freq
            x[col+'_'+'month'+'_cos']=np.cos(2*np.pi*month)/self.month_freq

            x[col+'_'+'month_day'+'_sin']=np.sin(2*np.pi*day_of_month)/self.month_day_freq
            x[col+'_'+'month_day'+'_cos']=np.cos(2*np.pi*day_of_month)/self.month_day_freq

            del x[col]
            # for features in self.feature_names:
            #     if col+'_weekday' in features:
            #         date=features.split('_')[1]
            #         x[features] = np.sin(2*np.pi *date/self.frequencies[date])
            #     else:
            #         del x[col]
        return x
    def get_feature_names(self):
        return self.feature_names
class TimeDifference(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names = ['dates_diff_min']
    def fit(self,x ,y=None):
        self.feature_names = x.columns
    def transform(self,x):
        diff = (x.iloc[:,0]-x.iloc[:,1]).dt.days
        return pd.DataFrame({'dates_diff_min': diff})
    def get_feature_names(self):
        return self.feature_names

class DataFrameImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names = []
        self.impute_dict = {}
    def fit(self, x, y=None):
        for col in x.columns:
            if x[col].dtype == 'O':
                self.impute_dict[col]='missing'
            else:
                self.impute_dict[col] = x[col].mean()
        return self
    def transform(self,x):
        for col in x.columns:
            col.fillna(self.impute_dict)
    def get_feature_names(self):
        return self.feature_names
    
class Get_Dummies(BaseEstimator, TransformerMixin):
    def __init__(self, frequency_cutoff=0):
        self.feature_names = []
        self.freequency_cutoff = frequency_cutoff
        self.var_cut_dict = {}
    def fit(self, x, y=None):
        data_cols = x.columns

        for cols in data_cols:
            k=x[cols].value_counts()
            if (k <= self.freequency_cutoff).sum() == 0:
                categories = k.index[:-1]
            else:
                categories = k.index[k>self.freequency_cutoff]
            self.var_cut_dict[col]=categories
        for col in self.var_cut_dict.keys():
            for cat in self.var_cut_dict[col]:
                self.feature_names.append(col+'_'+str(cat))
        
        return self
    def transform(self, x):
        dummy_data = x.copy()
        for col in self.var_cut_dict.keys():
            for cat in self.var_cut_dict[col]:
                name=col+'_'+str(cat)
                dummy_data[name] = (dummy_data[col]==cat).astype(int)
            del dummy_data[col]
        return dummy_data
    def get_feature_names(self):
        return self.feature_names
    
class pdPipeline(Pipeline):
    def get_feature_names(self):
        last_step = self.steps[-1][-1]

        return last_step.get_feature_names()
    



if __name__ == '__main__':
    df = pd.DataFrame({
        'dates': pd.date_range(start='1/1/2020', end='8/1/2021', periods=8),
        'dates_to': pd.date_range(start='1/1/2021', end='8/1/2022', periods=8)
        })
    obj = Convert_Datetime_Type(df)
    obj1 = ExtractDatetimeCyclic(df)
    obj2 = TimeDifference(df)
    # res = obj.convert_datetime_type()
    # res=obj1.cyclic_date_features()
    res = obj2.get_difference()
    print(res)

