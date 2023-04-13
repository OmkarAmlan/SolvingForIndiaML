#!/usr/bin/env python
# coding: utf-8

# In[128]:


import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_linear_regression
import math
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
import pickle


# In[16]:


df=pd.read_csv("bodyfat.csv")
df=df.iloc[:,1:]


# In[19]:


df


# In[126]:


column_headers = list(df.columns.values)
for i in column_headers:
    if i!="BodyFat":
        plt.scatter(df['BodyFat'],df[i])
        plt.title("Variation of bodyfat w.r.t. parameters")
        plt.xlabel("BodyFat")
        plt.ylabel("Parameters")


# In[47]:


df_x=df.iloc[:,1:]
df_y=df.iloc[:,0]


# In[63]:


x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=0.1)


# In[64]:


clf = linear_model.LinearRegression()
clf.fit(x_train, y_train)


# In[125]:


Age=int(input("Enter age "))
Weight=float(input("Enter weight in pounds "))
Height=float(input("Enter height in inches "))
Neck=float(input("Enter neck circumference in cms "))
Chest=float(input("Enter neck circumference in cms "))
Abdomen=float(input("Enter neck circumference in cms "))
Hip=float(input("Enter neck circumference in cms "))
Thigh=float(input("Enter neck circumference in cms "))
Knee=float(input("Enter neck circumference in cms "))
Ankle=float(input("Enter neck circumference in cms "))
Biceps=float(input("Enter neck circumference in cms "))
Forearm=float(input("Enter neck circumference in cms "))
Wrist=float(input("Enter neck circumference in cms "))
lst=[[Age,Weight,Height,Neck,Chest,Abdomen,Hip,Thigh,Knee,Ankle,Biceps,Forearm,Wrist]]
arr=np.array(lst)
y_pred=clf.predict(arr)
y_pred


# In[78]:


print(mean_squared_error(y_test, y_pred))
print(math.sqrt(mean_squared_error(y_test, y_pred)))
print(r2_score(y_pred,y_test))


# In[119]:


y_test


# In[130]:


pickle.dump(clf,open('model.pkl','wb'))

