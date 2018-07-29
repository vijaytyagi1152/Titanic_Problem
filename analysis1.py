import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


train=pd.read_csv("C:\\Users\\lenovo\\Desktop\\Dataset\\all\\train.csv")
test=pd.read_csv("C:\\Users\\lenovo\\Desktop\\Dataset\\all\\test.csv")
gender=pd.read_csv("C:\\Users\\lenovo\\Desktop\\Dataset\\all\\gender_submission.csv")

# fig=plt.Figure((30,30))
# fig.subplots(10,10)
# sns.countplot(x='Pclass',hue='Survived',data=train)

train['Age']=train['Age'].fillna(train['Age'].mean())
#print(train[train['Age'].isna()])
# fig.subplots(10,10)
# sns.distplot(train['Age'])
# plt.show()
gr=train.groupby('Sex')
print(gr.sum())
data_corr=train.iloc[:,1:].corr()
col=['Survived','Pclass','Age','SibSp','Parch','Fare']
sns.heatmap(data_corr,xticklabels=col,yticklabels=col)
plt.show()


# from sklearn.linear_model import LogisticRegression
# from sklearn.cross_validation import train_test_split
# pd.get_dummies(data)


# train_x, test_x, train_y, test_y = train_test_split(X,y,split=0.2,randome_state=10)

# reg=LogisticRegression()
# model=reg.fit(X,y)
# y_pred=model.predict(test_x)

# model.score(test_x,test_y)