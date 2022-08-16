import pandas as pd
import numpy as np
from datetime import date
from pandas.io import sql
from sqlalchemy import create_engine

class Two_Function:
    def two_func(self):
        d1 = pd.to_datetime('23-1-2020').toordinal()
        d2 = pd.to_datetime('23-12-2020').toordinal()

        df = pd.DataFrame({
            'date': [date.fromordinal(np.random.randint(d1, d2)) for i in range(100)],
            'sales': np.random.randint(100, 500, 100),
            'category': np.random.choice(['Apparels', 'Toys', "Consumables"], 100)
        })

        df['month'] = pd.DatetimeIndex(df['date']).month
        df1 = df.groupby(['month'])['sales'].mean()
        print(f'average sales across months = {df1}')

        # find category with minimum sales in second quarter
        # second-quarter is month 5 to 8
        df2 = df[(df['month'] >=4) & (df['month'] <=6)]
        # find minimum sales
        avg_df2 = df2.groupby(['category'])['sales'].sum()
        min_df2 = avg_df2.index[avg_df2==min(avg_df2)]
        # Get category for minimum sales
        print(f'category with minimum sales for the second quarter = {min_df2[0]}')


obj = Two_Function()
obj.two_func()

