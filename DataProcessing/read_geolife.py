# 파일명: read_geolife.py
import numpy as np
import pandas as pd
import glob
import os.path
import datetime
import os


def read_plt(plt_file):
    points = pd.read_csv(plt_file, skiprows=6, header=None,
                         parse_dates=[[5, 6]], infer_datetime_format=True)
    # 칼럼 이름 바꾸기
    points.rename(inplace=True, columns={'5_6': 'time', 0: 'lat', 1: 'lon', 3: 'alt'})
    # 안 쓰이는 칼럼 없애기
    points.drop(inplace=True, columns=[2, 4])
    # 경로 id 칼럼 추가
    trj_id = os.path.split(plt_file)[1].split('.')[0]
    points['trj_id'] = trj_id
    return points


mode_names = ['walk', 'bike', 'bus', 'car', 'subway', 'train', 'airplane', 'boat', 'run', 'motorcycle', 'taxi']
mode_ids = {s: i + 1 for i, s in enumerate(mode_names)}


# 아래 read_labels 부분은 각 좌표의 transport mode(도보, 택시, 버스 등등)과 관련된 부분.
# 그러나 모든 데이터에 이러한 label이 있는 것은 아니라서 label이 없는 데이터에는 그냥 0 을 넣는다.
def read_labels(labels_file):
    labels = pd.read_csv(labels_file, skiprows=1, header=None,
                         parse_dates=[[0, 1], [2, 3]],
                         infer_datetime_format=True, delim_whitespace=True)
    # 칼럼 이름 바꾸기
    labels.columns = ['start_time', 'end_time', 'label']

    # replace 'label' column with integer encoding
    labels['label'] = [mode_ids[i] for i in labels['label']]

    return labels


def apply_labels(points, labels):
    indices = labels['start_time'].searchsorted(points['time'], side='right') - 1
    no_label = (indices < 0) | (points['time'].values >= labels['end_time'].iloc[indices].values)
    points['label'] = labels['label'].iloc[indices].values
    points['label'][no_label] = 0


# 한 명의 유저 데이터 가져오는 부분
def read_user(user_folder):
    plt_files = glob.glob(os.path.join(user_folder, 'Trajectory', '*.plt'))
    df = pd.concat([read_plt(f) for f in plt_files])

    labels_file = os.path.join(user_folder, 'labels.txt')
    if os.path.exists(labels_file):
        labels = read_labels(labels_file)
        apply_labels(df, labels)
    else:
        df['label'] = 0

    return df


# 모든 유저 데이터 가져오는 부분
def read_all_users(folder):
    subfolders = os.listdir(folder)
    dfs = []
    for i, sf in enumerate(subfolders):
        print('[%d/%d] processing user %s' % (i + 1, len(subfolders), sf))
        df = read_user(os.path.join(folder, sf))
        df['user'] = int(sf)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

df = read_all_users(r'C:\Users\User\Desktop\Geolife Trajectories 1.3\Data')
df = df.reindex(columns=['time', 'lat', 'lon', 'alt', 'label', 'user', 'trj_id'])
df.to_csv('processed1.csv', index=False)
print(df)
