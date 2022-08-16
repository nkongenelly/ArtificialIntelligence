import pandas as pd

file = r"Edvancer/Exercises/data_handling/coupon_item.csv"
df = pd.read_csv(file)
print(df.head())

count_coupon_id = df.value_counts(['coupon_id']).reset_index()
count_coupon_id.columns = ['coupon_id', 'count_coupon_id']
# print('1. Count of how many times a coupon_id occurs in the dataset:')
# print(count_coupon_id.head())

unique_df = df.groupby(['coupon_id'])['item_id'].nunique().reset_index()
count_coupon_items = pd.merge(count_coupon_id, unique_df, on='coupon_id', how='left')
count_coupon_items.rename(columns={'item_id': 'coupon_unique_items_count'}, inplace=True)
del unique_df
# print(count_coupon_items.head())

count_categories_per_coupon = pd.crosstab(index=df['coupon_id'], columns=df['category'])
count_categories_per_coupon.columns = [x+'_counts' for x in list(count_categories_per_coupon.columns)]
count_categories_per_coupon_merge = pd.merge(count_coupon_items,count_categories_per_coupon, on='coupon_id', how='left') 
del count_categories_per_coupon
# print(count_categories_per_coupon_merge.head())

unique_categories_per_coupon = df.groupby(['coupon_id'])['category'].nunique().reset_index()
unique_categories_per_coupon.rename(columns={'category':'unique_categories'}, inplace=True)
unique_categories_per_coupon_merge = pd.merge(count_coupon_items, unique_categories_per_coupon, on='coupon_id', how='left')
del unique_categories_per_coupon
# print(unique_categories_per_coupon_merge.head())

brand_count_per_coupon = pd.DataFrame(df.groupby(['coupon_id'])['brand'].value_counts())
brand_count_per_coupon.columns=['brand_count']
brand_count_per_coupon.reset_index(inplace=True)
brand_count_per_coupon_max = brand_count_per_coupon.groupby(['coupon_id'])['brand'].nth(0).reset_index()
brand_count_per_coupon_max.rename(columns={'brand': 'max_brand_per_cat'}, inplace=True)
brand_count_per_coupon_max_merge = pd.merge(unique_categories_per_coupon_merge, brand_count_per_coupon_max, on='coupon_id', how='left')
del brand_count_per_coupon_max
# print(brand_count_per_coupon_max_merge.head())


# no. of brands per coupon which have frequency higher that 10% of that coupon count
brand_higher_by_ten = pd.DataFrame(df.groupby(['coupon_id'])['brand'].value_counts())
brand_higher_by_ten.columns=['brands_count']
brand_higher_by_ten.reset_index(inplace=True)

brand_higher_by_ten_merge1 = pd.merge(unique_categories_per_coupon_merge, brand_higher_by_ten, on='coupon_id', how='left')
brand_higher_by_ten_merge1['10%_coupon_count'] = brand_higher_by_ten_merge1['count_coupon_id'] * 0.1
brand_higher_by_ten_merge1_higher = brand_higher_by_ten_merge1[brand_higher_by_ten_merge1['brands_count'] > brand_higher_by_ten_merge1['10%_coupon_count']]
brand_higher_by_ten_merge1_count = pd.DataFrame(brand_higher_by_ten_merge1_higher.groupby(['coupon_id'])['brand'].nunique())
brand_higher_by_ten_merge1_count.columns = ['10%_count_brands']
brand_higher_by_ten_merge1_count.reset_index(inplace=True)

brand_higher_by_ten_merge1_count_merge = pd.merge(unique_categories_per_coupon_merge, brand_higher_by_ten_merge1_count, on='coupon_id', how='left')
del brand_higher_by_ten
del brand_higher_by_ten_merge1_count
print(brand_higher_by_ten_merge1_count_merge)