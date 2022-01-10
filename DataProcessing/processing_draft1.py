import numpy as np
import pandas as pd
import glob
import os.path
import datetime
import os

df = pd.read_csv(r'C:\Users\User\PycharmProjects\dataProcessing\processed1.csv')
df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M', errors='raise')
df['time'] = df['time'].dt.round('T')
# df.info()
# resampled = df.resample(rule='5T', on='time').first()
# print(df)

user = 0
trj = 0
resampled = []
temp = []

grouped = df.groupby(['user', 'trj_id'])
for key, group in grouped:
    i = 0
    j = 0
    print(key)
    temp.append(group.iloc[0])
    while i < len(group) and j < len(group):
        current = group.iloc[i, 0]
        next = group.iloc[j, 0]
        # print(current, ' ', next)

        time_diff = next - current
        # print(time_diff)
        if (time_diff) < datetime.timedelta(minutes=5):
            j += 1
        else:
            i = j
            temp.append(group.iloc[i])
    # print(temp)
    # print("* key", key)
    # print("* count", len(group))
    # print(group.head())
    # print('\n')

df1 = pd.DataFrame(temp)
df1.to_csv('processed2.csv', index=False)



