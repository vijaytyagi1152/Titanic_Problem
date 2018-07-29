import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.cross_validation import train_test_split

train=pd.read_csv("Titanic_Problem/train.csv")
test=pd.read_csv("Titanic_Problem/test.csv")
gender=pd.read_csv("Titanic_Problem/gender_submission.csv")
test['Survived']=-1
df=pd.concat([train,test],axis=0)
df=df.loc[:,['Age','Pclass','Sex','SibSp','Parch','Embarked','Fare','Survived']]
df['Age'].fillna(df['Age'].mean(),inplace=True)
df['Embarked'].replace(np.nan,'S',inplace=True)
#print(df.info())
#ig=plt.Figure()
#sns.countplot('SibSp',hue='Survived',data=df)
#sns.distplot(df['Age'])
#plt.show()
df['Pclass']=df['Pclass'].apply(lambda x:str(x))
dummy_df=pd.get_dummies(df)
# scaled_data=StandardScaler(dummy_df)
# print(scaled_data.)
print(dummy_df.columns)
#dummy_df.drop()
dummy_df.drop(columns=['Pclass_1','Sex_female','Embarked_C'],axis=1,inplace=True)
#print("1")
dummy_train=dummy_df[dummy_df['Survived']!=-1]
#print("2")
dummy_test=dummy_df[dummy_df['Survived']==-1]
#print("3")
dummy_test=dummy_test.loc[:,['Age','SibSp','Parch','Fare','Pclass_2','Pclass_1','Sex_male','Emabrked_Q','Embarked_C']]
#print("4")
dummy_test['Fare'].fillna(dummy_test['Fare'].mean(),inplace=True)
x_train,x_test,y_train,y_test=train_test_split(dummy_train.loc[:,['Age','SibSp','Parch','Fare','Pclass_2','Pclass_3','Sex_male','Emabrked_Q','Embarked_C']],dummy_train['Survived'],test_size=0.2,random_state=45)

logr=LogisticRegression()
print(x_train.info())
print("",end="\n\n")
print(y_train.info())

#rfecv=RFECV(logr)
model=logr.fit(x_train,y_train)
#model=logr.fit(dummy_train.loc[:,['Age','SibSp','Parch','Fare','Pclass_2','Pclass_1','Sex_male','Emabrked_Q','Embarked_C']])
#print(model.score(x_test,y_test))
ypred=model.predict(dummy_test)

import statsmodels.api as sm
l=sm.Logit(y_train,x_train)
smodel=l.fit()
print(smodel.summary())
import csv
f=open('submission.csv','w')
csvwrite=csv.writer(f)
for i in range (418):
    csvwrite.writerow([test.loc[i,'PassengerId'],ypred[i]])
    #print([test.loc[i,'PassengerId'],ypred[i]]])
f.close()
sns.relplot(x='Fare',y='Survived',hue='Survived',data=dummy_train)
plt.show()
