#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import glob
import os.path
import datetime
import os
import pymysql


# In[3]:


df = pd.read_csv('within_beijing.csv')
df.reset_index(drop=True, inplace=True)
print(df.head)


# In[5]:


df.columns


# In[6]:


df.index


# In[7]:


df.reset_index(inplace=True)


# In[8]:


df.head


# In[9]:


df.columns


# In[10]:


df.to_csv('within_beijing_indexed.csv', index=False)


# In[ ]:




