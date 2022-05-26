#!/usr/bin/env python
# coding: utf-8

# In[10]:


import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [12, 8]



# information of groups

group_names = ['Restaurant', 'Store', 'Parking', 'Lodging', 'Beauty_Salon','University',
              'Transit_Station','ATM','Hospital','Tourist_Attraction']

group_sizes = [2565,1567,481,467,306,279,237,200,192,112]




# Pie chart

plt.pie(group_sizes, 


        labels=group_names, 

        colors=group_colors, 

        autopct='%1.2f%%', # second decimal place

        shadow=True, 

        startangle=90,

        textprops={'fontsize': 14}) # text font size

plt.axis('equal') #  equal length of X and Y axis

plt.title('Number of locations by location type.', fontsize=20)

plt.show()


# In[16]:


import datetime
import requests
import pandas as pd
import numpy as np
from haversine import haversine
from datetime import timedelta

df=pd.read_csv('5min_seq.csv')
types_k=["Ac","Ap","Ag","Ax","Yx","Bx","Qx","Bs","Bi","Bt","Ba","Ce","Cr","Cw","Cx","Cm","Ch","Cs","Cv","Ct","Dx","Dr","Ex",
         "Fs","Fh","Fx","Gs","Gx","Hc","Hg","Hx","Ia","Jx","Nx","Wx","Xx","Lg","Lx","Mt","Mq","Mv","Mc","Mx","Nc",
         "Kx","Px","Ps","Ox","Po","Pr","Re","Rx","Sc","Ss","Sm","Sp","St","Sx","Sb","Ta","Ts","Tx","Tg","Ux","Vx","Zx"]

arr=df['sequence']
arr2=df['id']


            
for i in range(len(arr)):
    print('tid : '+str(arr2[i]))
    for j in range(len(types_k)):
        if types_k[j] in arr[i]:
            count=arr[i].count(types_k[j])
            print(types_k[j]+' : '+str(count))



# In[1]:


import datetime
import requests
import pandas as pd
import numpy as np
from haversine import haversine
from datetime import timedelta

df=pd.read_csv('5min_seq_trimmed_0301.csv')
types_k=["Ac","Ap","Ag","Ax","Yx","Bx","Qx","Bs","Bi","Bt","Ba","Ce","Cr","Cw","Cx","Cm","Ch","Cs","Cv","Ct","Dx","Dr","Ex",
         "Fs","Fh","Fx","Gs","Gx","Hc","Hg","Hx","Ia","Jx","Nx","Wx","Xx","Lg","Lx","Mt","Mq","Mv","Mc","Mx","Nc",
         "Kx","Px","Ps","Ox","Po","Pr","Re","Rx","Sc","Ss","Sm","Sp","St","Sx","Sb","Ta","Ts","Tx","Tg","Ux","Vx","Zx"]

arr=df['sequence']
arr2=df['id']
type_sum=0

for i in range(len(types_k)):
    for j in range(len(arr)):
        if types_k[i] in arr[j]:
            count=arr[j].count(types_k[i])
            type_sum+=count
    print(types_k[i]+' : '+str(type_sum))
    type_sum=0


# In[ ]:


import datetime
import requests
import pandas as pd
import numpy as np
from haversine import haversine
from datetime import timedelta

df=pd.read_csv('5min_seq.csv')
types_k=["Ac","Ap","Ag","Ax","Yx","Bx","Qx","Bs","Bi","Bt","Ba","Ce","Cr","Cw","Cx","Cm","Ch","Cs","Cv","Ct","Dx","Dr","Ex",
         "Fs","Fh","Fx","Gs","Gx","Hc","Hg","Hx","Ia","Jx","Nx","Wx","Xx","Lg","Lx","Mt","Mq","Mv","Mc","Mx","Nc",
         "Kx","Px","Ps","Ox","Po","Pr","Re","Rx","Sc","Ss","Sm","Sp","St","Sx","Sb","Ta","Ts","Tx","Tg","Ux","Vx","Zx"]

arr=df['sequence']
arr2=df['id']
arr3=df['user_id']
arr3=list(arr3)

uid=dict.fromkeys(arr3)
uid=list(uid)

tid=[]
seq=''

for i in range(len(uid)):
    tid.clear()
    seq=''
    print('uid - '+str(uid[i])+" : ")
    for j in range(len(arr3)):
        if arr3[j]==uid[i]:
            seq=seq+arr[j]
    for k in range(len(types_k)):
        if types_k[k] in seq:
            count=seq.count(types_k[k])
            print(types_k[k]+" : "+str(count)+" ")
    print('-------------------------')


# In[ ]:


import datetime
import requests
import pandas as pd
import numpy as np
import datetime
from haversine import haversine
from datetime import timedelta

df=pd.read_csv('5min_seq_0224.csv')

df['time_min']=pd.to_timedelta(df['time'])
df['min']=df['time_min'].dt.total_seconds().div(60).astype(int)

df.to_csv('5min_seq_0224.csv')

