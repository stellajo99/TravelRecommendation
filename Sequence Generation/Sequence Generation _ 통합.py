#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import datetime
import requests
import pandas as pd
import numpy as np
from haversine import haversine

df = pd.read_csv('5min_threshold_stop.csv')

arr2=df['dist']
dist_out=[]

# dist 100m 넘으면 제거
for i in range(len(arr2)):
    if(arr2[i]<0 or arr2[i]>100):
        dist_out.append(arr2[i])
        
mask=df['dist'].isin(dist_out)
df1=df[~mask]

df1['time']=-1
arr=df1['tid']
arr=list(arr)
tid_cnt=[]
tid_cnt=[0]*50728
tid_out=[]

tid=dict.fromkeys(arr)
tid=list(tid)

# 한 trj에 stop detection point가 3개 이하면 제거
for i in range(len(tid)):
    cnt=arr.count(tid[i])
    tid_cnt[i]=cnt
    if(tid_cnt[i]<4):
        tid_out.append(tid[i])


mask2=df1['tid'].isin(tid_out)
df2=df1[~mask2]

df2.to_csv('places_api_remove_0224.csv')


# In[ ]:


import datetime
import requests
import pandas as pd
import numpy as np
import math
from haversine import haversine

df = pd.read_csv('places_api_remove_0224.csv')

df['count']=-1

dataformat="%Y-%m-%d %H:%M:%S"

# 각 장소에 머무른 시간 구해서 5로 나눈 후 내림해서 추가
for i in range(len(df)):
    str_datetime1=df.iloc[i]['datetime']
    str_datetime2=df.iloc[i]['leaving_datetime']
    start=datetime.datetime.strptime(str_datetime1,dataformat)
    finish=datetime.datetime.strptime(str_datetime2,dataformat)
    time=(finish-start).seconds/60
    
    df.at[i,'time']=time
    df.at[i,'count']=math.floor(time/5)
    
df.to_csv('places_api_removeNtime_0224.csv')


# In[ ]:


# 시퀀스 만들기 전 각 장소에 고유 번호 붙이기
import datetime
import requests
import pandas as pd
import numpy as np
import math
from haversine import haversine

df = pd.read_csv('places_api_removeNtime_0224.csv')

df['seq']=''
df['sequence']=''

arr=df['place_id']
arr=list(arr)
places=dict.fromkeys(arr)
places=list(places)

arr2=df['type']
arr2=list(arr2)
types=dict.fromkeys(arr2)
types=list(types)
types.sort()

arr3=df['seq']

count=1

for i in range(len(types)):
    for j in range(len(places)):
        indexNum=arr.index(places[j])
        if arr2[indexNum]==types[i]:
            df.at[indexNum,'seq']=types[i]+str(count).zfill(4)
            count+=1
    print(types[i]+str(count-1))
    count=1   

for i in range(len(places)):
    for j in range(len(df)):
        if arr[j]==places[i]:
            indexNum=arr.index(places[i])
            arr3[j]=arr3[indexNum]
                
for i in range(len(df)):
    df.at[i,'sequence']=df['seq'][i]*df['count'][i]
                
df.to_csv('places_seq_0224.csv')


# In[ ]:


# 시퀀스 생성
import datetime
import requests
import pandas as pd
import numpy as np
from haversine import haversine

df1 = pd.read_csv('places_seq_0224.csv')
df2 = pd.read_csv('5min_simplasampled.csv')
df3 = pd.read_csv('5min_sequence.csv')

dataformat="%Y-%m-%d %H:%M"
dataformat2="%Y-%m-%d %H:%M:%S"

arr=df1['tid']
arr=list(arr)

arr2=df2['tid']
arr2=list(arr2)

arr3=df1['sequence']
df3['id']=''
df3['sequence']=''
df3['time']=0

tid=dict.fromkeys(arr)
tid=list(tid)

temp=[]
temp2=[]

time=[]
time2=[]
time3=[]
time_fin=[]
time_sub=0

sequence=''
seq=[]

for i in range(len(tid)):
    temp.clear()
    temp2.clear()
    time.clear()
    time_fin.clear()
    time2.clear()
    time3.clear()
    sequence=''
    for j in range(len(arr)):
        if(arr[j]==tid[i]):
            temp.append(j)
            
    print(temp)
    
    for k in range(len(arr2)):
        if(arr2[k]==tid[i]):
            temp2.append(k)
    
    print(temp2)
    
    for p in range(len(temp)):
        str_datetime1=df1.iloc[temp[p]]['datetime']
        str_datetime2=df1.iloc[temp[p]]['leaving_datetime']
        start=datetime.datetime.strptime(str_datetime1,dataformat)
        finish=datetime.datetime.strptime(str_datetime2,dataformat)
        time.append(start)
        time_fin.append(finish)
 

    for q in range(len(temp2)):
        str_datetime=df2.iloc[temp2[q]]['datetime']
        start=datetime.datetime.strptime(str_datetime,dataformat2)
        time2.append(start)

    time3=time+time2
    time3.sort()
    time_sub=time3[len(time3)-1]-time3[0]

    size=len(time3)
    t=0
    
    while t<size:
        if time3[t] in time2:
            if t==0:
                t+=1
            elif t==(len(time3)-1):
                t+=1
            elif time3[t-1] in time:
                if time3[t]<time_fin[time.index(time3[t-1])]:
                    time3.remove(time3[t])
                    size=size-1
                    t=i-1
                else:
                    t+=1
            else:
                t+=1
        elif time3[t] in time:
            t+=1
    
    for s in range(len(time3)):
        if time3[s] in time:
            indexNum=time.index(time3[s])
            sequence=sequence+arr3[temp[indexNum]]
        elif time3[s] in time2:
            sequence=sequence+'-'
    
    df3.at[i,'id']=tid[i]
    df3.at[i,'sequence']=sequence
    df3.at[i,'time']=time_sub
    
    print(sequence)
    print(str(i)+' '+str(tid[i])+' '+str(sequence))
    
df3.to_csv('5min_sequence_0224.csv')   


# In[ ]:


# 생성한 시퀀스에서 장소 type들을 두 개의 알파벳으로 축소
import datetime
import requests
import pandas as pd
import numpy as np
from haversine import haversine

df1 = pd.read_csv('5min_sequence_0224.csv')
df2 = pd.read_csv('places_seq_0224.csv')

arr1=df1['sequence']

arr2=df2['type']
arr2=list(arr2)
types=dict.fromkeys(arr2)
types=list(types)
types.sort()
types.remove('park')
types_k=["Ac","Ap","Ag","Ax","Yx","Bx","Qx","Bs","Bi","Bt","Ba","Ce","Cr","Cw","Cx","Cm","Ch","Cs","Cv","Ct","Dx","Dr","Ex","Fs","Fh","Fx","Gs","Gx","Hc","Hg","Hx","Ia","Jx","Nx","Wx","Xx","Lg","Lx","Mt","Mq","Mv","Mc","Mx","Nc","Px","Ps","Ox","Po","Pr","Re","Rx","Sc","Ss","Sm","Sp","St","Sx","Sb","Ta","Ts","Tx","Tg","Ux","Vx","Zx"]

for i in range(len(types)):
    for j in range(len(arr1)):
        text=arr1[j]
        text_mod=text.replace(types[i],types_k[i])
        arr1[j]=text_mod
 
df1.to_csv('5min_seqseq_0224.csv')


# In[ ]:


import datetime
import requests
import pandas as pd
import numpy as np
from haversine import haversine

df1 = pd.read_csv('5min_seqseq_0224.csv')

arr1=df1['sequence']

for i in range(len(arr1)):
    text=arr1[i]
    text_mod=text.replace('park','Kx')
    arr1[i]=text_mod
 
df1.to_csv('5min_seqseq_0224.csv')


# In[ ]:


# 생성한 sequence 파일에 user_id 추가
import datetime
import requests
import pandas as pd
import numpy as np
from haversine import haversine

df1 = pd.read_csv('5min_seqseq_0224.csv')
df2 = pd.read_csv('places_seq_0224.csv')

df1['user_id']=''
arr1=df2['uid']
arr2=df1['id']
arr3=df2['tid']
arr1=list(arr1)
arr2=list(arr2)
arr3=list(arr3)

for i in range(len(arr2)):
    indexNum=arr3.index(arr2[i])
    uid=arr1[indexNum]
    df1.at[i,'user_id']=uid

df1.to_csv('5min_seq_0224.csv')

