import pandas as pd
import numpy as np
import matplotlib

class PythonPipelineOne:
    def __init__(self,columns):
        self.columns = columns
    def convert_datetime_type(self):
        print(self.columns['dates'])
        self.columns['dates'] = pd.to_datetime(self.columns['dates'])
        self.columns['dates_to'] = pd.to_datetime(self.columns['dates_to'])
        return self.columns

class ExtractDatetime(PythonPipelineOne):
    def __init__(self, columns):
        super().__init__(columns)
        self.convert_datetime_type()
    def extract_dates(self):
        self.columns['weekday'] = self.columns['dates'].dt.weekday #Monday = 0 and Sunday = 6
        self.columns['month'] = self.columns['dates'].dt.month #January = 1 and December = 12
        self.columns['day_of_month'] = self.columns['dates'].dt.day
        # self.columns["min_from_midnight"]=np.linspace(0,24*60-1,8, dtype=int)
        # self.columns['minutes'] = self.columns['dates'].dt.minute
        # self.cyclic_date_features()
        return self.columns
    def cyclic_date_features(self):
        self.extract_dates()
        # sin((2*pie*x)/max(x)) and cos((2*pie*x)/max(x))
        self.columns['weekday_sin'] = np.sin(2*np.pi *self.columns['weekday'] / 7)
        self.columns['weekday_cos'] = np.cos(2*np.pi *self.columns['weekday'] / 7)

        self.columns['month_sin'] = np.sin(2*np.pi *self.columns['month'] / 12)
        self.columns['month_cos'] = np.cos(2*np.pi *self.columns['month'] / 12)

        self.columns['day_of_month_sin'] = np.sin(2*np.pi *self.columns['day_of_month'] / 31)
        self.columns['day_of_month_cos'] = np.cos(2*np.pi *self.columns['day_of_month'] / 31)
        return self.columns
class TimeDifference(PythonPipelineOne):
    def __init__(self,columns):
        super().__init__(columns)
        self.convert_datetime_type()
    def get_difference(self):
        self.columns['dates_diff_min'] = (self.columns['dates_to']-self.columns['dates'])/pd.Timedelta(minutes=1)
        return self.columns



df = pd.DataFrame({
    'dates': pd.date_range(start='1/1/2020', end='8/1/2021', periods=8),
    'dates_to': pd.date_range(start='1/1/2021', end='8/1/2022', periods=8)
    })
obj = PythonPipelineOne(df)
obj1 = ExtractDatetime(df)
obj2 = TimeDifference(df)
# res = obj.convert_datetime_type()
# res=obj1.cyclic_date_features()
res = obj2.get_difference()
print(res)

