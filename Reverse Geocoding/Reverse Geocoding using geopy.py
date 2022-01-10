#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install reverse_geocoder')


# In[6]:


import reverse_geocoder as rg
coordinates = (39.984702,116.318417)
location=rg.search(coordinates)


# In[10]:


print('latitude: ', location[0]['lat'])
print('longitude: ', location[0]['lon'])
print('name: ', location[0]['name'])


# In[9]:


location


# In[18]:


from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="myGeolocator")
location=geolocator.reverse("39.984702, 116.318417",language='en')


# In[19]:


print("latitude: ",location.latitude)
print("longitude: ",location.longitude)


# In[20]:


location.raw


# In[16]:


location.raw['address']


# In[5]:


import pandas as pd

df=pd.read_csv(r'C:\Users\minsk\20min_threshold.csv')
df.head()


# In[6]:


df["geom"] = df["lat"].map(str)+','+df["lng"].map(str)
df["geom"][0]


# In[34]:


from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="myGeolocator")


for i in range(1,19503):
    location=geolocator.reverse(df["geom"][i-1],language='en')
    print((str)(i)+': '+location.raw['display_name'])
    
df.to_csv('reverse_result.csv')
    


# In[ ]:





# In[ ]:




