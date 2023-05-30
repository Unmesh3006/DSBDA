#!/usr/bin/env python
# coding: utf-8

# In[3]:


import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.corpus import stopwords


# In[ ]:


#not removed punctuations


# In[4]:


nltk.download('averaged_perceptron_tagger')


# In[6]:


text='Blue Campaign is a national public awareness campaign designed to educate the public, law enforcement, and other industry partners to recognize the indicators of human trafficking, and how to appropriately respond to possible cases.'


# In[7]:


tokens=nltk.sent_tokenize(text)
tokens


# In[8]:


tokenwords=nltk.word_tokenize(text)
tokenwords


# In[9]:


from nltk.stem import PorterStemmer, WordNetLemmatizer


# In[10]:


import pandas as pd
import numpy as np


# In[12]:


stop_words = set(stopwords.words('english'))
stop_words


# In[15]:


#words which are not included in stopwords
wordt=[wordt for wordt in tokenwords if wordt not in stop_words]
wordt


# In[16]:


#pos tagging
tagged=nltk.pos_tag(tokenwords)
for tag in tagged:
    print(tag)


# In[17]:


ps=PorterStemmer()
stemmed={word:ps.stem(word) for word in tokenwords}
for pair in stemmed.items():
    print('{0}---{1}'.format(pair[0],pair[1]))


# In[19]:


lem=WordNetLemmatizer()
wordlem={word: lem.lemmatize(word) for word in tokenwords}
for x in wordlem.items():
    print('{0}-->{1}'.format(x[0],x[1]))


# In[36]:


symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
for i in symbols:
    tokenwords = np.char.replace(tokenwords, i, ' ')


# In[20]:


#term frequency and document frequency


# In[29]:


def arr_convert_1d(arr):
    arr = np.array(arr)
    arr = np.concatenate( arr, axis=0 )
    arr = np.concatenate( arr, axis=0 )
    print('testing')
    print(arr)
    return arr


# In[30]:


cos = []
def cosine(trans):
    cos.append(cosine_similarity(trans[0], trans[1]))


# In[31]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[32]:


def tfidf(str1, str2):
    vect = TfidfVectorizer()
    vect.fit(tokenwords)
    corpus = [str1,str2]
    trans = vect.transform(corpus)
    cosine(trans)
    return convert()


# In[33]:


def convert():
    dataf = pd.DataFrame()
    lis2 = arr_convert_1d(cos)
    dataf['cos_sim'] = lis2
    return dataf


# In[34]:


str1 = 'Blue'
str2 = 'law'
newData = tfidf(str1,str2);
print(newData);


# In[ ]:




