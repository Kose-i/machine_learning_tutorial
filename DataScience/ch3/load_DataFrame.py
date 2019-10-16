import pandas as pd
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt

#student_data_math = pd.read_csv('student-mat.csv')
#student_data_math.head()

student_data_math = pd.read_csv('student-mat.csv', sep=';')
student_data_math.head()
print(student_data_math.info())

print(student_data_math.groupby('sex')['age'].mean())

print("平均値:{0}".format(student_data_math['absences'].mean()))
print("中央値:{0}".format(student_data_math['absences'].median()))
print("最頻値:{0}".format(student_data_math['absences'].mode()))
print("分散  :{0}".format(student_data_math['absences'].var()))
print("偏差  :{0}".format(student_data_math['absences'].std()))

print("要約統計量")
print(student_data_math['absences'].describe())
print("四方位範囲")
print(student_data_math['absences'].describe()[6] - student_data_math['absences'].describe()[4])

# 箱ひげ図:G1
plt.boxplot(student_data_math['G1'])
plt.grid(True)

# 箱ひげ図:G1, G2, G3
plt.boxplot(student_data_math['G1'], student_data_math['G2'], student_data_math['G3'])
plt.grid(True)
plt.show()

# 変動係数
#print(student_data_math['absences'].std() / student_data_math['absences'].mean())
print(student_data_math.std() / student_data_math.mean())

# 散布図
plt.plot(student_data_math['G1'], student_data_math['G3'],'o')
plt.ylabel('G3 grade')
plt.xlabel('G1 grade')
plt.grid(True)
plt.show()

# 共分散行列
print(np.conv(student_data_math['G1'], student_data_math['G3']))

# 相関係数
print(sp.stats.pearsonr(student_data_math['G1'], student_data_math['G3']))

# 相関行列
print(np.corrcoef([student_data_math['G1'], student_data_math['G3']]))

sns.pairplot(student_data_math[['Dalc','Walc','G1','G3']])
plt.grid(True)
plt.show()

##Chapter3-4
