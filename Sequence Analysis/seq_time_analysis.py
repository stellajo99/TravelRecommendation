#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import glob
import os.path
import datetime
import os
import ast


# In[5]:


df = pd.read_csv('5min_seq_trimmed.csv')


# In[6]:


# row 생략 없이 출력
pd.set_option('display.max_rows', None)
# col 생략 없이 출력
pd.set_option('display.max_columns', None)


# In[33]:


trimmed_all = []
duration_all = []
for i in range(len(df)):
    trimmed = df['trimmed'].iloc[i]
    duration = df['durations'].iloc[i]
    trimmed = ast.literal_eval(trimmed)
    duration = ast.literal_eval(duration)
    for j in range(len(trimmed)):
        trimmed_all.append(trimmed[j])
        duration_all.append(duration[j] * 5)

dict_df = pd.DataFrame({"location":trimmed_all,
                       "duration":duration_all})


# In[34]:


print(dict_df)


# In[35]:


dict_df.groupby(['location'], as_index=False).mean().sort_values(by=['duration'], ascending = False)


# In[36]:


grouped = dict_df.groupby('location')
grouped


# In[ ]:





# In[37]:


# 지나치게 큰 값 -> 중간값과 비교 필요
# ex) 2449번째 sequence에 한 번 등장하는 Sx1522
grouped.mean().sort_values(by=['duration'], ascending = False)


# In[38]:


# 중간값, 등장 횟수를 고려해 봤을 때 한 번 등장했는데 오래 머물러서 큰 값이 나옴
grouped.median().sort_values(by=['duration'], ascending = False)


# In[39]:


# 장소별 등장 횟수
grouped.count().sort_values(by=['duration'], ascending = False)


# In[40]:


types = {'Rx': 0, 'Sx': 0, 'Px': 0, 'Lx': 0, 'Bs': 0, 'Ux': 0, 'Sc': 0, 'Tx': 0, 'Ax': 0, 'Cr': 0, 'Bx': 0, 'Hx': 0, 'Lg': 0, 'Ta': 0, 'Kx': 0, 'Yx': 0, 'Nx': 0, 'Cs': 0, 'Re': 0, 'Hc': 0, 'Tg': 0, 'St': 0, 'Ox': 0, 'Ss': 0, 'Wx': 0, 'Hg': 0, 'Qx': 0, 'Mx': 0, 'Gs': 0, 'Gx': 0, 'Sm': 0, 'Bt': 0, 'Fx': 0, 'Xx': 0, 'Nc': 0, 'Ac': 0, 'Po': 0, 'Ia': 0, 'Mt': 0, 'Ce': 0, 'Cx': 0, 'Ap': 0, 'Ex': 0, 'Vx': 0, 'Mv': 0, 'Bi': 0, 'Zx': 0, 'Sp': 0, 'Ch': 0, 'Dx': 0, 'Mc': 0, 'Cm': 0, 'Fh': 0, 'Ts': 0, 'Ag': 0, 'Fs': 0, 'Jx': 0, 'Ba': 0, 'Cw': 0, 'Cv': 0, 'Ct': 0, 'Dr': 0, 'Mq': 0, 'Ps': 0, 'Pr': 0, 'Sb': 0}

for i in range(len(df)):
    locs = df.iloc[i]['places']
    locs = ast.literal_eval(locs)
    for j in range(len(locs)):
        abbr = locs[j][0:2]
        types[abbr] += 1
    print(locs)
print("Done!") 


# In[41]:


print(types)


# In[50]:


cnt_all = []
type_all = []

for i in range(len(df)):
    # str type인 seq_list를 파싱 후 list로 만듦
    seq_list = df.iloc[i]['seq_list']
    seq_list = seq_list[1:len(seq_list) - 1]
    seq_list = seq_list.split(", ")   
    temp_list = []
    for j in seq_list:
        temp_list.append(j[1:7])   
    
    # trimmed list 생성하는 부분
    pointer = temp_list[0][0:2]
    
    cnt_list = []
    type_list = []
    type_list.append(pointer)
    cnt = 1
    
    for j in range(1, len(temp_list) + 1):
        # 비교해서 같으면 패스, 다르면 리스트에 넣음
        if j == len(temp_list):
            cnt_list.append(cnt)    
            break
        if pointer == temp_list[j][0:2]: 
            cnt += 1
            continue
        else:
            pointer = temp_list[j][0:2]
            type_list.append(pointer)
            cnt_list.append(cnt)
            cnt = 1
    cnt_all += cnt_list
    type_all += type_list
    # print(i+2, cnt_list, type_list)

duration = [0 for i in range(len(cnt_all))]

for k in range(len(cnt_all)):
    duration[k] = cnt_all[k] * 5 
print(type_all)
print(cnt_all)
print(duration)


# In[51]:


type_df = pd.DataFrame({"type":type_all,
                       "duration":duration})


# In[52]:


print(type_df)


# In[53]:


type_df.groupby(['type'], as_index=False).mean().sort_values(by=['duration'], ascending = False)


# In[ ]:




