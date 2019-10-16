from pandas import DataFrame
import pandas as pd

data1 = {
  'id': ['100','101','102','103','104','106','108','110','111','113'],
  'city': ['Tokyo','Osaka','Kyoto','Hokkaido','Tokyo','Tokyo','Osaka','Kyoto','Hokkaido','Tokyo'],
  'birth_year':[1990,1989,1992,1997,1982,1991,1988,1990,1995,1981],
  'name':['Hiroshi','Akiko','Yuki','Satoru','Steeve','Mituru','Aoi','Tarou','Suguru','Mitsuo']
}
df1 = DataFrame(data1)
print(df1)

data2 = {
  'id':['100','101','102','105','107'],
  'math':[50,43,33,76,98],
  'english':[90,30,20,50,30],
  'sex':['M','F','F','M','M'],
  'index_num':[0,1,2,3,4]
}
df2 = DataFrame(data2)
print(df2)

# 内部結合
print('内部結合')
print(pd.merge(df1, df2, on='id'))

# 全結合
print('全結合')
print(pd.merge(df1, df2, how='outer'))

# 左外部結合
print('左外部結合')
print(pd.merge(df1, df2, how='left'))

# 縦結合
print('縦結合')
print(pd.concat([df1,df2]))


target_dataframe = pd.concat([df1, df2])
print('target_table')
print(target_dataframe)
# リストワイズ削除
print('リストワイズ削除')
print(target_dataframe.dropna())

# ペアワイズ削除
print('ペアワイズ削除')
print(target_dataframe[[0,1]].dropna())

# fillnaで埋める
print('fillnaで埋める')
print(target_dataframe.fillna(0))

# 前の値で埋める
print('前の値で埋める')
print(target_dataframe.fillna('ffill'))

# 平均値で埋める
print('平均値で埋める')
print(target_dataframe.fillna(target_dataframe.mean()))
