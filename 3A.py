#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


df=pd.read_csv("nba.csv")


# In[4]:


df.head()


# In[5]:


df.describe


# In[6]:


df.info


# In[7]:


df.info()


# In[8]:


df['Height'].value_counts()


# In[9]:


df['Weight'].value_counts()


# In[10]:


heightgrp=df.groupby(df['Height'])


# In[11]:


heightgrp.get_group('6-0')


# In[12]:


weightgrp=df.groupby(df['Weight'])


# In[13]:


weightgrp


# In[14]:


type(weightgrp)


# In[15]:


weightgrp.dtypes


# In[16]:


bins=[19,25,31,36,40]
labels=['19-24','25-30','31-35','36-40']
df['Agegrp']=pd.cut(df['Age'],bins=bins,labels=labels,right=False)


# In[17]:


df.head()


# In[18]:


df.info()


# In[19]:


df['Agegrp'].value_counts()


# In[20]:


df.groupby('Agegrp')['Salary'].max()


# In[21]:


df.groupby('Agegrp')['Salary'].min()


# In[22]:


df.groupby('Agegrp')['Salary'].std()


# In[23]:


df.groupby('Agegrp')['Salary'].mean()


# In[24]:


df.groupby('Agegrp')['Salary'].count()


# In[25]:


salary_list=list(df.groupby('Agegrp')['Salary'])
salary_list


# In[26]:


df


# In[28]:


df.head(458)


# In[ ]:




