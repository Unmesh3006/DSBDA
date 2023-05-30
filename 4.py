#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install seaborn


# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pylot as plt
import seaborn as sns
df=pd.read_csv("titanic.csv")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


pip install matlpotlib


# In[6]:


import matplotlib.pyplot as plt


# In[9]:


import pandas as pd
import numpy as np

import seaborn as sns
df=pd.read_csv("titanic.csv")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


df.head()


# In[11]:


sns.distplot(df['Fare'])


# In[12]:


sns.distplot(df['Fare'],kde=False)


# In[13]:


sns.distplot(df['Fare'],kde=False,bins=10)


# In[14]:


df.info()


# In[15]:


sns.jointplot(x='Age',y='Fare',data=df)


# In[17]:


sns.jointplot(x='Age',y='Fare',data=df,kind='hex')


# In[18]:


sns.jointplot(x='Age',y='Fare',data=df,kind='reg')


# In[20]:


sns.jointplot(x='Pclass',y='Fare',data=df,kind='hex')


# In[22]:


sns.pairplot(df)


# In[25]:


sns.pairplot(data=df,hue='Sex')


# In[26]:


sns.pairplot(data=df,kind='hist')


# In[30]:


sns.rugplot(df[ 'Fare' ])


# In[ ]:




