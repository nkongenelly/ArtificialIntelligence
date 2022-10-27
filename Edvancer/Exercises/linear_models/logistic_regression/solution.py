import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.metrics import fbeta_score
from sklearn.base import BaseEstimator, TransformerMixin


from operator import itemgetter

class Logistic_Regression:
    def __init__(self, file):
        self.df = pd.read_excel(file, skiprows=1)
    
        # ignore ID column
        self.df = self.df.drop('ID', axis=1)

        # transform the right-skewed columns
        self.right_skewed_cols = self.df.iloc[:,self.df.columns.get_loc('BILL_AMT1'): self.df.columns.get_loc('PAY_AMT6')]

        self.target_index = self.df.columns.get_loc('default payment next month')
        self.cat_columns = ['SEX', 'EDUCATION', 'PAY_0','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

    def set_X_train(self,param):
        self.X_train = param
    
    def set_X_test(self,param):
        self.X_test = param
    
    def set_y_train(self,param):
        self.y_train = param
    
    def set_y_test(self,param):
        self.y_test = param

    def transform_right_skewed(self):
        for col in self.right_skewed_cols.columns:
            if (self.df[col].values < 0 ).any() == False:
                self.df[col] = np.log10(self.right_skewed_cols[col])
        return self.df


    def draw_point_plt_for_columns(self, data):
        sns.pairplot(data, hue='default payment next month')
        plt.show()

    # def draw_histogram():
    #     sns.

    def impute_missing(self):
        for col in self.df.columns:
            if self.df[col].isnull().sum() > 0:
                if self.df[col].dtype == 'O':
                    self.df[col] = self.df[col].fillna(self.df[col].mode())
                else:
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
        return self.df

    def convert_to_int(self):
        for col in self.right_skewed_cols:
            if self.df[col].dtype == 'float64':
                # self.df[col] = pd.to_numeric(self.df[col], errors='coerce').astype(int)
                # self.df[col] =  np.floor(pd.to_numeric(self.df[col], errors='coerce')).astype('Int64')
                self.df[col] = self.df[col].astype('int')

        return self.df

    def get_dummyData(self, cutoff=0):
        dummify = categorical_dummies(cutoff, file)
        dummify.fit()
        self.df = dummify.transform()

    def cross_validate(self, model, X, y, cv_no):
        scores = cross_val_score(model, X, y, cv=cv_no)
        print(f'Expected score = {scores.mean()}')

    def grid_Search(self, model, param_grid, cv_no, X, y):
        search = GridSearchCV(estimator = model, param_grid=param_grid, cv=cv_no, refit=True)
        search.fit(X, y)

        print(f'best params = {search.best_params_}') 
        # print(f'best no. of features = {search.best_params_.get_params()}')
        print(f'model accuracy = {search.best_score_}')
        return search.best_params_

    def best_features(self, model, param_grid, best_scoring, X, y):
        fold = KFold(n_splits=5, shuffle=True, random_state=100)

        feature_search = GridSearchCV(estimator=RFE(model), param_grid=param_grid, scoring=best_scoring, cv=fold, return_train_score= True)
        feature_search.fit(X, y)
        feature_no = feature_search.best_params_
        print(f'----------------{feature_no}------------------')
        # print(feature_search.cv_results_)

        return feature_no

    def RFE_model(self, model, feature_no, X, y):
        rfe_object = RFE(estimator=model, n_features_to_select = feature_no)
        # X = X.loc[:,['PAY_0', 'PAY_2', 'PAY_AMT2']]
        rfe_object.fit(X, y)

        self.print_rankings_coef(rfe_object)

        # Predict
        y_pred = rfe_object.predict(X)
        # Predict probability 
        y_pred_proba = rfe_object.predict_proba(X)[:,1]

        # calculate perfomance
        print('--------------------------Perfomance without optimum threshold---------------------------')
        self.calculate_perfomance(y, y_pred, y_pred_proba)

        print(list(zip(y, y_pred_proba))[:10])
        print('-----------correlation matrix-------')
        # print(X.corr())

        # Calculate best cutoff
        y_pred_cutoff = cat_model.calculate_best_cutoff(y_pred_proba, y)


        # calculate perfomance
        print('--------------------------Perfomance WITH optimum threshold---------------------------')
        cat_model.calculate_perfomance(y, y_pred, y_pred_cutoff)

        ## EXTRA: Confusion Matrix
        cm = confusion_matrix(y, y_pred_cutoff) # rows = truth, cols = prediction
        df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
        plt.figure(figsize = (10,7))
        sns.set(font_scale=1.4)
        sns.heatmap(df_cm, annot=True, fmt='g')
        print("Test Data Accuracy: %0.4f" % accuracy_score(y, y_pred)) 
        print('-----------confusion matrix-------')
        print(df_cm)

    def print_rankings_coef(self, rfe_object):
        
        print(rfe_object.ranking_)
        print('-------------rfe.support columns-------------')
        print(X.columns[rfe_object.support_])
        best_columns = X.columns[rfe_object.support_]
        column = X.columns.to_list()

        print('-------------coefficients -------------')
        print(rfe_object.estimator_.coef_)
        rankings = []
        for m,n, in list(sorted(zip(rfe_object.ranking_, column), key = itemgetter(0))):
            print(m,n)
            for o,c in list(zip(rfe_object.estimator_.coef_[0], column)):
                if c == n:
                    rankings += [[m,n, o]]
            # if m == feature_no:
            #     break
        rankings_pd = pd.DataFrame(rankings, columns=['ranking', 'column', 'coefficient'])
        print(rankings_pd)
    def calculate_perfomance(self, y, y_pred, y_pred_proba):
        # Perfomance
        accuracy = accuracy_score(y, y_pred)    # = 0.813143
        precision = precision_score(y, y_pred)    # = 0.693738
        f1 = f1_score(y, y_pred)    # = 0.363194
        recall = recall_score(y, y_pred)    # = 0.245988
        roc_auc = roc_auc_score(y, y_pred_proba)  # = 0.704876 / 0.522225

        perfomance_score = pd.DataFrame([[accuracy,precision,f1,recall, roc_auc]], columns=['accuracy','precision','f1','recall', 'roc_auc'])

        print('perfomance score for l1 penalty = ')
        print(perfomance_score)

    def calculate_best_cutoff(self, y_pred_proba, y):
        precisions, recalls, thresholds = precision_recall_curve(y, y_pred_proba)

        # fscores = 0 if ((precisions + recalls) == 0)  else ((2 * precisions * recalls) / (precisions + recalls))
        fscores = (2 * precisions * recalls) / (precisions + recalls)
        optimal_idx = np.argmax(fscores)
        
        threshold =  thresholds[optimal_idx]
        fsscores = fscores[optimal_idx]
        print('--------threshold---------')
        print(threshold)
        print('--------fsscores---------')
        print(fsscores)

        prediction = (y_pred_proba >= threshold).astype(int)

        return prediction

class categorical_dummies(BaseEstimator, TransformerMixin, Logistic_Regression):
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

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  
# def threshold_from_desired_precision(self, X, y, desired_precision=0.9):
#     y_scores = LogisticRegression.predict_proba(self, X)[:, 1]
#     precisions, recalls, thresholds = precision_recall_curve(y, y_scores)

#     desired_precision_idx = np.argmax(precisions >= desired_precision)
    
#     return thresholds[desired_precision_idx], recalls[desired_precision_idx]

# def threshold_from_desired_recall(self, X, y, desired_recall=0.9):
#     y_scores = LogisticRegression.predict_proba(self, X)[:, 1]
#     precisions, recalls, thresholds = precision_recall_curve(y, y_scores)

#     desired_recall_idx = np.argmin(recalls >= desired_recall)
    
#     return thresholds[desired_recall_idx], precisions[desired_recall_idx]

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 
# draw_point_plt_for_columns(df.iloc[:,[0,1,2,3,target_index ]])

# file = r'D:\AI_ML\AI\Machine Learning in Python\data\data\default of credit card clients.xls'
file = r'/mnt/d/AI_ML/AI/Machine Learning in Python/data/data/default of credit card clients.xls'

cat_model = Logistic_Regression(file)
# df = transform_right_skewed(df)
cat_model.impute_missing()
cat_model.convert_to_int()
cat_model.get_dummyData(500)

X = cat_model.df.drop('default payment next month', axis=1)
y = cat_model.df['default payment next month']
print('---x shape---- = ', X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

# Set the data into the class
cat_model.set_X_train(X_train)
cat_model.set_X_test(X_test)
cat_model.set_y_train(y_train)
cat_model.set_y_test(y_test)

# Scaling the data
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train.values), index=X_train.index, columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test.values), index=X_test.index, columns=X_test.columns)

model_lr = LogisticRegression()

cat_model.cross_validate(model_lr, X_train, y_train, 5)   # 0.7787000000000001

param_grid_model = {'penalty':['l1', 'l2'], 'C':np.logspace(-3,3,7)}
param_grid_features = {'n_features_to_select': np.arange(0,25,3)}
# best_params = cat_model.grid_Search(model_lr, param_grid_model, 5, X_train, y_train)  # best params = {'C': 100.0, 'penalty': 'l2'}, model accuracy = 0.7788333333333334

model_lr_best = LogisticRegression(C = 100, penalty='l2')
# model_lr_best = LogisticRegression(C = best_params['C'], penalty=best_params['penalty'])

# feature selection
# TODO : Test with different scoring and knowhow success is measured in the different acoring

scoring_f1 = 'f1'
scoring_roc_auc = 'roc_auc'
# feature_calc = cat_model.best_features(model_lr_best, param_grid_features, scoring_roc_auc, X_train, y_train)
feature_no =  24 #feature_calc['n_features_to_select']

# Logistic regression with RFE
cat_model.RFE_model(model_lr_best, feature_no, X_train, y_train)
# accuracy    precision       f1        recall
# 0.812476    0.69911      0.389647    0.27009
