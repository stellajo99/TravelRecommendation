import numpy as np
import pandas as pd
import glob
import os.path
import datetime
import os

df = pd.read_csv('within_beijing_indexed.csv')
df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S', errors='raise')
# df['datetime'] = df['datetime'].dt.round('T')
# df.info()
# resampled = df.resample(rule='5T', on='time').first()
print(df.head)

# dftemp = []
grouped = df.groupby(['user', 'tid'])

for key, group in grouped:
    temp = []
    i = 0
    j = 1
    print(key)
    while i < len(group) and j < len(group):
        current = group.iloc[i, 1]
        next = group.iloc[j, 1]
        # print(current, ' ', next)
        time_diff = next - current
        # print(time_diff)
        if (time_diff.seconds / 3600) > 3:
            # print(time_diff)
            temp.append(j)
            i += 1
            j += 1
        else:
            i += 1
            j += 1
    # print(temp)
    if (len(temp) == 0):
        continue
    elif (len(temp) == 1):
        start = temp[0]
        end = len(group)
        date = group.iloc[start]['datetime']
        pid = date.strftime('%Y%m%d%H%M%S')
        for u in range(start, end):
            index = group.iloc[u]['index']
            df.iat[index, 7] = pid
            print(df.iloc[index]['index'], df.iloc[index]['tid'])
    else:
        k = 0
        while k <= len(temp) - 1:
            if (k == len(temp) - 1):
                start = temp[k]
                end = len(group)
            else:
                start = temp[k]
                end = temp[k + 1]
            date = group.iloc[start]['datetime']
            pid = (date.strftime('%Y%m%d%H%M%S'))
            for t in range(start, end):
                # group.iloc[t]['trd_id'] = pid
                index = group.iloc[t]['index']
                df.iat[index, 7] = pid
                print(df.iloc[index]['index'], df.iloc[index]['tid'])
            k += 1

            # group.iloc[j]['tid'] = group.iloc[j]['datetime']

# print(dftemp)
# for i in range(len(dftemp)):
#     index = dftemp[i][0]
#     df.iat[index, 6] = dftemp[i][1]
#     print(df.iloc[dftemp[i][0]]['tid'])

df.to_csv('trj_separation.csv', index=False)




