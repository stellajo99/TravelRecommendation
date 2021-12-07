#!/usr/bin/env python
# coding: utf-8

# In[1]:


#샘플 데이터 만드는 코드
import random
import sys
from tensorflow import keras
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import string
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt

def generate_time_series():
    string_pool=["A","B","C","D","E","F"]
    char_to_index=dict((char,index) for index, char in enumerate(string_pool))
    index_to_char={}
    
    for key,value in char_to_index.items():
        index_to_char[value]=key
    
    array=[[0 for col in range(30)] for row in range(100)]
    for i in range(100):
        for j in range(30):
            a=random.choice(string_pool)
            num=char_to_index[a]
            array[i][j]=num
        array[i].sort()
        
    for i in range(50):
        array[i].reverse()
    arr=np.array(array)
    np.random.shuffle(arr)
    arr=np.array(arr)
    return arr

series=generate_time_series()
print(series+1)


# In[9]:


#Clustering
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from matplotlib import cm
import warnings as w
w.filterwarnings('ignore')

#위의 코드로 생성한 샘플 데이터를 엑셀 파일로 저장해서 데이터로 사용
ds=pd.read_csv(r'C:\Users\minsk\ttopic.csv')
x=ds.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]].values

wcss=[]
for i in range(1,10):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=100)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

#군집 품질을 확인할 수 있는 왜곡 점수
print('왜곡 : %.2f'%kmeans.inertia_)

#Elbow Method
plt.plot(range(1,10),wcss)
plt.xlabel('Number of clusters')
plt.ylabel('Within Cluster sum of squares')
plt.show()

#Clustering - 4개의 cluster로 나눈 경우
kmeans=KMeans(n_clusters=4, init='k-means++', random_state=100)
y_kmeans=kmeans.fit_predict(x)
print(y_kmeans)

#군집 품질을 확인할 수 있는 실루엣 그래프
cluster_labels=np.unique(y_kmeans)
n_clusters=cluster_labels.shape[0]
silhouette_vals=silhouette_samples(x,y_kmeans,metric='euclidean')
y_ax_lower,y_ax_upper=0,0
yticks=[]

for i,c in enumerate(cluster_labels):
    c_silhouette_vals=silhouette_vals[y_kmeans==c]
    c_silhouette_vals.sort()
    y_ax_upper+=len(c_silhouette_vals)
    color=cm.jet(float(i)/n_clusters)
    plt.barh(range(y_ax_lower,y_ax_upper),c_silhouette_vals,height=1.0,edgecolor='none',color=color)
    yticks.append((y_ax_lower+y_ax_upper)/2.)
    y_ax_lower+=len(c_silhouette_vals)
    
silhouette_avg=np.mean(silhouette_vals)
plt.axvline(silhouette_avg,color="red",linestyle="--")
plt.yticks(yticks,cluster_labels+1)
plt.ylabel('Clusters')
plt.xlabel('Silhouette coefficient')
plt.tight_layout()
plt.show()

clu_1=[]
clu_2=[]
clu_3=[]
clu_4=[]

for i in range(99):
    if(y_kmeans[i]==0):
        clu_1.append(i+1)
        
    elif y_kmeans[i]==1:
        clu_2.append(i+1)
        
    elif y_kmeans[i]==2:
        clu_3.append(i+1)
        
    else:
        clu_4.append(i+1)
        
print(' ')
print('[ Cluster 1~4로 나뉜 경로들 ]')
print('###################################################################################################################################')        
print('Cluster 1에 속한 경로 :',clu_1)
print('Cluster 2에 속한 경로 :',clu_2)
print('Cluster 3에 속한 경로 :',clu_3)
print('Cluster 4에 속한 경로 :',clu_4)
print('###################################################################################################################################')
print(' ')

#Clustering - Elbow method를 이용해서 나온 2개의 군집으로 나눌 경우
kmeans=KMeans(n_clusters=2, init='k-means++', random_state=100)
y_kmeans=kmeans.fit_predict(x)
print(y_kmeans)

#군집 품질을 확인할 수 있는 실루엣 그래프
cluster_labels=np.unique(y_kmeans)
n_clusters=cluster_labels.shape[0]
silhouette_vals=silhouette_samples(x,y_kmeans,metric='euclidean')
y_ax_lower,y_ax_upper=0,0
yticks=[]

for i,c in enumerate(cluster_labels):
    c_silhouette_vals=silhouette_vals[y_kmeans==c]
    c_silhouette_vals.sort()
    y_ax_upper+=len(c_silhouette_vals)
    color=cm.jet(float(i)/n_clusters)
    plt.barh(range(y_ax_lower,y_ax_upper),c_silhouette_vals,height=1.0,edgecolor='none',color=color)
    yticks.append((y_ax_lower+y_ax_upper)/2.)
    y_ax_lower+=len(c_silhouette_vals)
    
silhouette_avg=np.mean(silhouette_vals)
plt.axvline(silhouette_avg,color="red",linestyle="--")
plt.yticks(yticks,cluster_labels+1)
plt.ylabel('Clusters')
plt.xlabel('Silhouette coefficient')
plt.tight_layout()
plt.show()



# In[ ]:




