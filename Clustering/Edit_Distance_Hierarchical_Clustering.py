#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 랑데부 논문 참고, min 대신 max 사용
# indel은 0, 변경엔 같으면 +1, 다르면 +0

import datetime
import requests
import pandas as pd
import numpy as np
import scipy.spatial.distance as ssd
from haversine import haversine
from tqdm import trange, notebook


df=pd.read_csv('5min_seq_trimmed_0301.csv')
df['trimmed'] = df.trimmed.apply(lambda x: x[1:-1].split(','))
arr=df.trimmed

df2=pd.read_csv('places_info_id_0224.csv',encoding='cp949')
tyid=df2['type_id']
tyid=list(tyid)
lat=df2['lat']
lat=list(lat)
lng=df2['lng']
lng=list(lng)

def levenshtein(arr1, arr2): 
    size_x = len(arr1) + 1   
    size_y = len(arr2) + 1
    matrix = np.zeros ((size_x+1, size_y+1))
    for x in range(size_x): 
        matrix [x+1, 0] = x
    for y in range(size_y):
        matrix [0, y+1] = y
        
    for x in range(2, size_x): 
        for y in range(2, size_y): 
            seq1=arr1[x-2]
            seq2=arr2[y-2]
            
            if (x-2)==0:
                seq1=seq1[1:7]
            else:
                seq1=seq1[2:8]

            if (y-2)==0:
                seq2=seq2[1:7]
            else:
                seq2=seq2[2:8]

            type_id1=seq1
            type_id2=seq2
            
            gps1=(lat[tyid.index(type_id1)],lng[tyid.index(type_id1)])
            gps2=(lat[tyid.index(type_id2)],lng[tyid.index(type_id2)])
            
            dist=haversine(gps1,gps2,unit='m')
            #print(dist)
            #print(type_id1+type_id2)
            distNum=0

            if dist<500:  
                matrix [x,y] = max(
                    matrix[x-1, y] ,#문자삭제
                    matrix[x-1, y-1]+1 ,   #문자변경
                    matrix[x, y-1],    #문자삽입
                    0
                )

            else :
                matrix [x,y] = max(   
                    matrix[x-1,y] ,
                    matrix[x-1,y-1],
                    matrix[x,y-1],
                    0
                )


            
    #print(matrix)  
    return (matrix[size_x - 1, size_y - 1])  # matrix에서 제일 오른쪽아래값을 출력해줌 


a=[]
for i in notebook.tqdm(range(len(arr))):
    result=[]
    for j in range(len(arr)):
        # print(levenshtein(arr[i],arr[j]))
        result.append(levenshtein(arr[i],arr[j]))
    #print(result)
    a.append(result)

print(a)

a=np.array(a)

distA_norm = a

for i in notebook.tqdm(range(len(arr))):
    for j in range(len(arr)):
        distA_norm[i][j]=(a[i][j]/(min(len(arr[i]),len(arr[j]))-1))
        
print(distA_norm)

for i in notebook.tqdm(range(len(arr))):
    for j in range(len(arr)):
        distA_norm[i][j] = 1-distA_norm[i][j]
        
print(distA_norm)

distArray=ssd.squareform(distA_norm)
print(distArray)


# In[ ]:


import scipy.spatial.distance as ssd
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import cophenet
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import cut_tree
from scipy.spatial.distance import pdist
import numpy as np


result = linkage(distArray, 'ward')
print(result)
# Ward Linkage — Uses the analysis of variance method to determine the distance between clusters
# 클러스터 내의 분산을 최소화하기 위한 방법으로, 
# 가중 거리를 계산하면서 군집 결합도를 빼주는 방식으로 두 군집 간의 거리를 정의한다. 
# 군집 중심 사이의 가중 제곱 거리를 최소화하는 방식으로 계산한다.

plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
#plt.axhline(y=326.5, c='k')
dendrogram(
    result,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=7.,  # font size for the x axis labels
)
# plt.axhline(y=25, c='k')
plt.show()


# In[ ]:


#Cluster 개수를 변경해가며 silhouette coefficient 분석

avg_sil=[]
for i in range(1, len(arr)):
    cluster_info = cut_tree(result, i).flatten()
    
    print("Cluster # : "+str(i))
    
    for i in range(len(cluster_info)):
        cluster_info[i]=cluster_info[i]+1

        
        #df['cluster'] = cluster_info
    
    from sklearn import metrics
    nodes = cluster_info
    a = ssd.squareform(distArray)
    average_score = metrics.silhouette_score(a, nodes, metric='precomputed')
    print(average_score)
    avg_sil.append(average_score)
    a = ssd.squareform(distArray)
    score_samples = metrics.silhouette_samples(a, nodes, metric='precomputed')
    df['silhouette_coeff'] = score_samples

    df.groupby('cluster')['silhouette_coeff'].mean()


# In[ ]:


import matplotlib.pyplot as plt

avg_sil_ = np.mean(avg_sil)

plt.plot(range(6,300), avg_sil, 'bx-')
plt.axhline(y=avg_sil_, color='r', linestyle='dotted')
plt.xlabel('# of Clusters')
plt.ylabel('Silhouette Coefficient')
plt.title('Silhouette')
plt.show()


# In[ ]:


print(avg_sil_)
print(np.min(avg_sil))
print(np.max(avg_sil))


# In[ ]:


# intra-cluster distance
sum=0
for i in range(1,(len(arr)-1),1):
    for j in range(i+1,len(arr),1):
        a1 = index[i]
        a2 = index[j]
        sum+=distA_norm[a1][a2]
        
sum = sum*2/((len(arr)-1)*len(arr))
print(sum)


# In[ ]:


intra_clu=[]
for m in range(1, len(arr)+1):
    cluster_info = cut_tree(result, m).flatten()
    
    print("Cluster # : "+str(m))
    
    for i in range(len(cluster_info)):
        cluster_info[i]=cluster_info[i]+1
    #df['cluster'] = cluster_info
    #df.to_csv('max_cluster_info_7.csv')
    #print(cluster_info)

    clu_num=cluster_info
    clu_num=list(clu_num)
    #print(clu_num)
    total_sum=0
        
    for j in range(1,m+1):
        index_=[]
        for k in range(300):
            if clu_num[k]==j:
                index_.append(k)
            
            sum=0
        #print(index_)
        for p in range(0,len(index_),1):
            for q in range(p+1, len(index_),1):
                a1=index_[p]
                a2=index_[q]
                sum+=distA_norm[a1][a2]
        if len(index_)>1:
            sum=sum*2/((len(index_)-1)*len(index_))
        else:
            sum=0
        #sum=sum*2/((len(index_)-1)*len(index_))
        #print(str(j)+' '+str(sum))
        total_sum+=sum
    total_sum=total_sum/m
    intra_clu.append(total_sum)
    print("Intra-cluster Distance when Cluster # is "+str(m)+" : "+str(total_sum))

print(intra_clu)


# In[ ]:


import matplotlib.pyplot as plt

avg = np.mean(intra_clu)
print(avg)

plt.plot(range(1,301), intra_clu, 'bx-')
plt.axhline(y=avg, color='r', linestyle='dotted')
plt.xlabel('# of Clusters')
plt.ylabel('Similarity among Trajectories')
plt.title('Intra-cluster Distance')
plt.show()


# In[ ]:


print(avg)
print(np.min(intra_clu))
print(np.max(intra_clu))

