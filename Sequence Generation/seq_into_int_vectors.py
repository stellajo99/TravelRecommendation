#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import numpy as np


# In[36]:


df = pd.read_csv('locations_limited_to_30_types.csv')
places = []


# In[37]:


for i in range(len(df)):
    cnt = df['count'][i]
    for j in range(1,cnt+1):
        place = df['abbr'][i]+str(j).zfill(4)
        places.append(place)


# In[38]:


places.insert(0, '-')
places.insert(1, 'start')
places.insert(2, 'end')
print(len(places))
print(" ")
print(places)


# In[39]:


places_dict = {string : i+1 for i, string in enumerate(places)}
print(places_dict)


# In[40]:


inverse_places = {v: k for k, v in places_dict.items()}
print(inverse_places)


# In[41]:


places = np.array(places)


# In[42]:


print(places)


# In[43]:


df_places = pd.DataFrame(places)


# In[44]:


df_places


# In[45]:


df_places.rename( columns={0:'place'}, inplace=True)


# In[46]:


df_places


# In[47]:


df_places['int'] = 0
for k in range(len(df_places)):
    df_places['int'][k] = k+1


# In[48]:


df_places


# In[49]:


df_places.to_csv('place_int_pair_0324.csv')


# In[50]:


df2 = pd.read_csv('5min_seq_trimmed_clst.csv')


# In[51]:


df2.rename( columns={'seq_list':'seq_list_short'}, inplace=True)


# In[52]:


# sequence -> list
cnt = 0
df2['seq_list_long'] = 'a'
for i in df2['sequence']:
    temp_list = []
    j = 0
    while j < len(i):
        if i[j] == '-':
            temp_list.append(i[j])
            j = j + 1
        else:
            temp_list.append(i[j:j+6])
            j = j + 6
    # print(trimmed)
    df2['seq_list_long'].iloc[cnt] = temp_list
    cnt = cnt + 1


# In[56]:


from ast import literal_eval
df2['trimmed'] = df2['trimmed'].apply(lambda x: literal_eval(str(x)))


# In[58]:


df2['trimmed']


# In[63]:


# whole sequence to int list
cnt = 0
df2['seq_int_vec'] = 'a'
df2['trimmed_seq_int_vec'] = 'a'

for i in range(len(df2)):
    seq = df2['seq_list_long'][i]
    trimmed_seq = df2['trimmed'][i]
    
    temp_list = []
    trimmed_temp_list = []
    
    for j in range(len(seq)):
        temp_list.append(places_dict[seq[j]])
        
    for k in range(len(trimmed_seq)):
        new = trimmed_seq[k].replace("'", "").strip()
        trimmed_temp_list.append(places_dict[new])
        
    df2['seq_int_vec'].iloc[cnt] = temp_list
    df2['trimmed_seq_int_vec'].iloc[cnt] = trimmed_temp_list
    
    cnt += 1


# In[64]:


df2['seq_int_vec']
length = np.array([len(x) for x in df2['seq_int_vec']])
df2['length_vec'] = length


# In[65]:


df2


# In[66]:


df2.to_csv('5min_seq_intVec_added_0325.csv')


# In[67]:


length = np.array([len(x) for x in df2['seq_int_vec']])


# In[68]:


print(np.mean(length), np.median(length), np.max(length))


# In[69]:


import matplotlib.pyplot as plt
plt.hist(length)
plt.xlabel('length')
plt.ylabel('frequency')
plt.show()


# In[ ]:




