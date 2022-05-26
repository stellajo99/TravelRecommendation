#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import glob
import os.path
import datetime
import os


# In[2]:


# row 생략 없이 출력
# pd.set_option('display.max_rows', None)
# col 생략 없이 출력
# pd.set_option('display.max_columns', None)


# In[3]:


df = pd.read_csv('5min_seq_mean.csv')


# In[4]:


df['trimmed'] = 'error'
df['trimmed_cnt'] = 0
df['durations'] = 'hey'

for i in range(len(df)):
    # str type인 seq_list를 파싱 후 list로 만듦
    seq_list = df.iloc[i]['seq_list']
    seq_list = seq_list[1:len(seq_list) - 1]
    seq_list = seq_list.split(", ")   
    temp_list = []
    for j in seq_list:
        temp_list.append(j[1:7])   
    
    # trimmed list 생성하는 부분
    pointer = temp_list[0]
    
    cnt_list = []
    final_list = []
    final_list.append(pointer)
    cnt = 1
    
    for j in range(1, len(temp_list) + 1):
        # 비교해서 같으면 패스, 다르면 리스트에 넣음
        if j == len(temp_list):
            cnt_list.append(cnt)    
            break
        if pointer == temp_list[j]: 
            cnt += 1
            continue
        else:
            pointer = temp_list[j]
            final_list.append(pointer)
            cnt_list.append(cnt)
            cnt = 1
    print(i+2, cnt_list)
    df['trimmed'].iloc[i] = final_list
    df['trimmed_cnt'].iloc[i] = len(final_list)
    df['durations'].iloc[i] = cnt_list


# In[5]:


df.to_csv('5min_seq_trimmed.csv', index=False)


# In[ ]:




