import pandas as pd
from one import *
from sklearn.pipeline import FeatureUnion

train_file = r'Edvancer/Exercises/python_pipelines/Property_train.csv'
test_file = r'Edvancer/Exercises/python_pipelines/Property_test_share.csv'

df_train = pd.read_csv(train_file)
df_test = pd.read_csv(test_file)

# df_train.head()
# df_train.info(verbose=True, show_counts=True)
# """
# A - Impute missing - ['PriceIndex9','Zip', 'InsurancePremiumIndex', 'PropertyAge', 'SubModel', 'NormalisedPopulation', 'BuildYear', '']

# B - Convert_To_Numeric - ['PriceIndex8','PriceIndex7', 'PriceIndex4', 'PriceIndex1', 'PriceIndex6', 'Channel', 'PlotType', 'Architecture', 'PriceIndex2', 'PriceIndex3', 'PriceIndex5', '']

# C - Categorical_Dummies - ['InteriorsStyle','Material', 'Agency', 'AreaIncomeType', 'EnvRating', 'ExpeditedListing', 'PRIMEUNIT', 'Region', 'Facade', 'State', 'RegionType' ]

# D - Convert_To_Date - ['ListDate']

# E - Cyclic_Features - ['ListDate']

# """
# remove columns with all null
# print(df_train.head())
for cols in df_train.columns:
    if int(df_train[cols].isnull().sum()) == df_train.shape[0] :
        print(list(zip([cols, int(df_train[cols].isnull().sum()), df_train.shape[0]])))
        del df_train[cols]

date_col=['ListDate']
target=['Junk']

cats_col=df_train.select_dtypes(['object']).columns
cats_col=[temp for temp in cats_col if 'Price' not in temp]
if date_col[0] in cats_col:
    cats_col.remove(date_col[0])
# cats_col
num_cols = df_train.columns
# num_cols
num_cols=[temp for temp in num_cols if temp not in date_col+cats_col+target]
# num_cols
if 'Zip' in num_cols:
    num_cols.remove('Zip')
cats_col.append('Zip')
# df_train.shape
if 'Zip' in df_train:
    df_train['Zip'].isnull().sum()

p1=pdPipeline([
    ('date_select', Var_Selector(date_col)),
    ('convert_to_datetime', Convert_Datetime_Type()),
    ('cyclic_features', ExtractDatetimeCyclic())
])

p2=pdPipeline([
    ('cat_select',Var_Selector(cats_col)),
    ('missing_trt',DataFrameImputer()),
    ('create_dummies',Get_Dummies(500))
    ])

p3=pdPipeline([
    ('num_cols',Var_Selector(num_cols)),
    ('convert_to_numeric', Convert_To_numeric()),
    ('missing_trt',DataFrameImputer())
])

data_pipe=FeatureUnion([
    ('data_pipe', p1),
    ('cat_pipe', p2),
    ('num_pipe', p3)
])
# print(df_train.columns)
try:
    data_pipe.fit(df_train)
    x_train = pd.DataFrame(data=data_pipe.transform(df_train),columns=data_pipe.get_feature_names())
    x_test = pd.DataFrame(data=data_pipe.transform(df_test),columns=data_pipe.get_feature_names())
    
except Exception as e:
    print('error = ', e)