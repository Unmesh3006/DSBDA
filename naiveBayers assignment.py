#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df=pd.read_csv("iris.csv")


# In[5]:


df


# In[6]:


df.head()


# In[7]:


df.dtypes


# In[9]:


df.isnull().sum()


# In[10]:


sns.boxplot(x='variety',y='sepal.length',data=df)


# In[11]:


sns.boxplot(x='variety',y='sepal.width',data=df)


# In[13]:


sns.countplot('variety',data=df)


# In[14]:


df.info()


# In[15]:


X=df[['sepal.length','sepal.width','petal.length','petal.width']]
y=df['variety']


# In[16]:


print(X)


# In[17]:


print(y)


# In[26]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2,shuffle=True)


# In[27]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[29]:


from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(X_train,y_train)


# In[30]:


y_predict=model.predict(X_test)
y_predict


# In[31]:


model.score(X_test, y_test)


# In[32]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# In[34]:


cm=confusion_matrix(y_test,y_predict)
displ=ConfusionMatrixDisplay(confusion_matrix=cm)
print(cm)


# In[36]:


displ.plot()
plt.show()


# In[38]:


def get_cm(y_test,y_pre):
    cm=confusion_matrix(y_test,y_pre)
    return(cm[0][0],cm[0][1],cm[1][0],cm[1][1])

TP, FP, FN, TN = get_cm(y_test, y_predict)
print("TP: ", TP)
print("FP: ", FP)
print("FN: ", FN)
print("TN: ", TN)


# In[ ]:


#acurracy,recall..voh sab karna hai

