import pandas as pd
import numpy as np
import matplotlib
import seaborn  



train=pd.read_csv("C:\\Users\\lenovo\\Desktop\\Dataset\\all\\train.csv")
test=pd.read_csv("C:\\Users\\lenovo\\Desktop\\Dataset\\all\\test.csv")
gender=pd.read_csv("C:\\Users\\lenovo\\Desktop\\Dataset\\all\\gender_submission.csv")

# print(train.info(),end="\n\n")
# print(test.info(),end="\n\n")
# print(gender.info())

print(train['Embarked'].mode())
print(train['Embarked'].fillna('S',inplace=True))
print(train['Name'][train.Embarked.isna()])
train['Title']=train['Name'].apply(lambda x:x.split(',')[1].split()[0])
print(pd.crosstab(train['Sex'],train['Embarked']))
