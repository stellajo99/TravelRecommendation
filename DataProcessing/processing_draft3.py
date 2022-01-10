import skmob
import pandas as pd
from skmob.preprocessing import filtering

tdf = skmob.TrajDataFrame.from_file(r'C:\Users\User\PycharmProjects\dataProcessing\processed1.csv', user_id='user_id', trajectory_id='trj_id', latitude='lat', longitude='lon', datetime='time')

print(tdf)
ftdf = filtering.filter(tdf, max_speed_kmh=500.)
print(ftdf.parameters)
n_deleted_points = len(tdf) - len(ftdf) # number of deleted points
print(n_deleted_points)