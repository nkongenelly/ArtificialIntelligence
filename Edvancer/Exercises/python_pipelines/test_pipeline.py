import warnings
import pandas as pd 
import numpy as np
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from test import *

warnings.filterwarnings('ignore')

train_file=r'Edvancer/Exercises/python_pipelines/loan_data_train.csv'
test_file=r'Edvancer/Exercises/python_pipelines/loan_data_test.csv'

ld_train=pd.read_csv(train_file)
ld_test=pd.read_csv(test_file) 

# ld_train.info()
# print(ld_train.sample(10))

p1=pdPipeline([
    ('var_select',Var_Selector(['Amount.Requested','Amount.Funded.By.Investors','Open.CREDIT.Lines','Revolving.CREDIT.Balance'])),
    ('convert_to_numeric',Convert_to_Numeric()),
    ('missing_trt',Impute_Missing())
])

p2=pdPipeline([
    ('var_select',Var_Selector(['Debt.To.Income.Ratio'])),
    ('string_clean',String_Replacement(replace_it='%',replace_with='')),
    ('convert_to_numeric',Convert_to_Numeric()),
    ('missing_trt',Impute_Missing())
])

p3=pdPipeline([
    ('var_select',Var_Selector(['Loan.Length', 'Loan.Purpose','State','Home.Ownership',
                               'Employment.Length'])),
    ('missing_trt',Impute_Missing()),
    ('create_dummies',Categorigal_Dummies(20))
])

p4=pdPipeline([
    ('var_select',Var_Selector(['Monthly.Income','Inquiries.in.the.Last.6.Months'])),
    ('missing_trt',Impute_Missing())
])

p5=pdPipeline([
    ('var_select',Var_Selector(['FICO.Range'])),
    ('custom_fico',Fico_separation()),
    ('missing_trt',Impute_Missing())
])

data_pipe=FeatureUnion([
    ('obj_to_num',p1),
    ('dtir',p2),
    ('obj_to_dum',p3),
    ('num',p4),
    ('fico',p5)
])

data_pipe.fit(ld_train)

len(data_pipe.get_feature_names())

data_pipe.transform(ld_train).shape

x_train=pd.DataFrame(data=data_pipe.transform(ld_train),
                    columns=data_pipe.get_feature_names())
                    

x_test=pd.DataFrame(data=data_pipe.transform(ld_test),
                    columns=data_pipe.get_feature_names())

x_train.shape

x_test.shape

x_train.columns

x_test.columns

