import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

class Var_Selector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self,x,y=None):
        return self
    def transform(self,x):
        return x[self.feature_names]
    def get_feature_names(self):
        return self.feature_names

class Impute_Missing(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.impute_dict={}
        self.feature_names=[]
    def fit(self,x,y=None):
        self.feature_names=x.columns

        for col in x.columns:
            if x[col].dtype=='O':
                self.impute_dict[col]='missing'
            else:
                self.impute_dict[col]=x[col].median()
        return self
    def transform(self,x,y=None):
        return x.fillna(self.impute_dict)
    def get_feature_names(self):
        return self.feature_names

class Convert_to_Numeric(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names=[]
    def fit(self,x,y=None):
        self.feature_names=x.columns
        return self
    def transform(self,x):
        for col in x.columns:
            x[col]=pd.to_numeric(x[col],errors='coerce')
        return x
    def get_feature_names(self):
        return self.feature_names

class Categorigal_Dummies(BaseEstimator, TransformerMixin):
    def __init__(self, freq_cutoff=0):
        self.freq_cutoff = freq_cutoff
        self.feature_names = []
        self.var_cut_dict = {}
    def fit(self, x, y=None):
        data_cols = x.columns

        for col in data_cols:
            k=x[col].value_counts()

            if(k<=self.freq_cutoff).sum() == 0:
                cats=k.index[:-1]
            else:
                cats=k.index[k>self.freq_cutoff]
            self.var_cut_dict[col]=cats

        for col in self.var_cut_dict.keys():
            for cat in self.var_cut_dict[col]:
                self.feature_names.append[col+'_'+cat]
        
        return self

    def transform(self,x,y=None):
        dummy_data=x.copy()

        for col in self.var_cut_dict.keys():
            for cat in self.var_cut_dict[col]:
                name=col+'_'+cat
                dummy_data[name]=(dummy_data[col]==cat).astype(int)

            del dummy_data[col]
        return dummy_data
    
    def get_feature_names(self):
        return self.feature_names

class String_Replacement(BaseEstimator, TransformerMixin):
    def __init__(self, replace_it='',replace_with=''):
        self.replace_it=replace_it
        self.replace_with=replace_with
        self.feature_names=[]
    def fit(self,x,y=None):
        self.feature_names = x.columns
        return self
    def transform(self, x):
        for col in x.columns:
            x[col]=x[col].str.replace(self.replace_it, self.replace_with)
        return x
    def get_feature_names(self):
        return self.feature_names

class Fico_separation(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names = ['fico']
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        k=x('FICO.Range').str.split("-", expand=True).astype(float)
        fico = 0.5*(k[0] + k[1])
        return pd.Dataframe({'fico':fico})
    def get_feature_names(self):
        return self.feature_names

class pdPipeline(Pipeline):
    def get_feature_names(self):
        # print(self.steps)
        last_step = self.steps[-1][-1]

        return last_step.get_feature_names()

# p1=pdPipeline([
#     ('var_select',Var_Selector(['Amount.Requested','Amount.Funded.By.Investors','Open.CREDIT.Lines','Revolving.CREDIT.Balance'])),
#     ('convert_to_numeric',Convert_to_Numeric()),
#     ('missing_trt',Impute_Missing())
# ])
# data_pipe=FeatureUnion([
#     ('obj_to_num',p1)
# ])

# train_file=r'Edvancer/Exercises/python_pipelines/loan_data_train.csv'

# ld_train=pd.read_csv(train_file)

# data_pipe.fit(ld_train)
# res = data_pipe.get_feature_names()
# print(res)