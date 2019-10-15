import pandas as pd
from pandas import Series, DataFrame

# Series
sample_pandas_data = pd.Series([0,10,20,30,40,50,60,70,80,90])
print(sample_pandas_data)

sample_pandas_index_data = pd.Series([0,10,20,30,40,50,60,70,80,90], index=['a','b','c','d','e','f','g','h','i','j'])
print(sample_pandas_data)

print("データの値:{0}".format(sample_pandas_index_data.values))
print("インデックスの値:{0}".format(sample_pandas_index_data.index))

# How to use 'DataFrame'
attri_data1 = {'ID':['100','101','102','103','104'],
               'City':['Tokyo','Osaka','Kyoto','Hokkaido','Tokyo'],
               'Birth_year':[1990, 1989, 1992, 1997,1982],
               'Name':['Hiroshi','Akiko','Yuki','Satoru','Steve']}
attri_data_frame1 = DataFrame(attri_data1)
print(attri_data_frame1)

attri_data_frame_index1 = DataFrame(attri_data1, index=['a','b','c','d','e'])
print(attri_data_frame_index1)

# 行列の操作
print(attri_data_frame1.T)
print(attri_data_frame1.Birth_year)
print(attri_data_frame1[['ID','Birth_year']])

# データの抽出
print(attri_data_frame1[attri_data_frame1['City']=='Tokyo'])
print(attri_data_frame1[attri_data_frame1['City'].isin(['Tokyo','Osaka'])])

# データの削除と結合
print(attri_data_frame1)
attri_data_frame1.drop(['Birth_year'],axis=1)
print(attri_data_frame1)

attri_data2 = {'ID':['100','101','102','105','107'],
               'Math':[50,43,33,76,98],
               'English':[90, 30, 20, 50,30],
               'Sex':['M','F','F','M','M']}
attri_data_frame2 = DataFrame(attri_data2)
print(attri_data_frame2)

print(pd.merge(attri_data_frame1, attri_data_frame2)) # 内部結合

# グループの集計
print(attri_data_frame2.groupby('Sex')['Math'].mean())
