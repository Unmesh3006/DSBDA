#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt


# In[2]:


import pandas as pd
import numpy as np

import seaborn as sns
df=pd.read_csv("titanic.csv")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df.head()


# In[4]:


sns.barplot(x= 'Sex' , y= 'Age' , data=df)


# In[5]:


sns.countplot(x= 'Sex' , data=df)


# In[6]:


sns.boxplot(x= 'Sex' , y= 'Age' , data=df)


# In[8]:


sns.countplot(x='Sex', data=df, hue='Survived')


# In[9]:


sns.boxplot(df['Age'])


# In[10]:


sns.boxplot(x='Sex', y='Age', data=df,hue="Survived")


# In[11]:


sns.violinplot(x='Sex', y='Age', data=df)


# In[12]:


sns.violinplot(x='Sex', y='Age', data=df, hue='Survived', split=True)


# In[13]:


sns.stripplot(x='Sex', y='Age', data=df, hue='Survived',split=True)


# In[ ]:




