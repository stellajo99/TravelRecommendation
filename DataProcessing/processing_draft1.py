# 파일명: read_geolife.py
import numpy as np
import pandas as pd
import glob
import os.path
import datetime
import os

pd.options.display.max_rows

df = pd.read_csv('processed1.csv')

i = 0
idx = 0
resampled = []
temp = []

while i < 182:
    condition1 = (df['user'] == idx)
    df1 = df[condition1]
    while df1.loc[idx]['trj_id'] == df1.loc[idx + 1]['trj_id']:
        temp.append(df1.loc[idx])
        idx += 1
    pd.concat(temp)
    print(temp)


