import numpy as np
import pandas as pd
import glob
import os.path
import datetime
import os

# df1 = pd.read_csv('processed1.csv')
# df1['time'] = pd.to_datetime(df1['time'], format='%Y-%m-%d %H:%M', errors='raise')
# df1['time'] = df1['time'].dt.round('T')
# df1.info()
# print('\n')

df2 = pd.read_csv('processed2.csv')
df2['time'] = pd.to_datetime(df2['time'], format='%Y-%m-%d %H:%M', errors='raise')
df2['time'] = df2['time'].dt.round('T')
print(df2)





