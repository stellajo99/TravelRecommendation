import skmob
import pandas as pd
from skmob.preprocessing import detection

tdf = skmob.TrajDataFrame.from_file('processed3.csv', user_id='user', trajectory_id='tid', latitude='lat', longitude='lng', datetime='datetime')
# print(tdf.head())

stdf1 = detection.stops(tdf, stop_radius_factor=0.5, minutes_for_a_stop=5.0, spatial_radius_km=0.2, leaving_time=True)
stdf2 = detection.stops(tdf, stop_radius_factor=0.5, minutes_for_a_stop=20.0, spatial_radius_km=0.2, leaving_time=True)
# print(stdf.head())

df1 = pd.DataFrame(stdf1)
df1.to_csv('5min_threshold.csv', index=False)

df2 = pd.DataFrame(stdf2)
df2.to_csv('20min_threshold.csv', index=False)

print('Points of the original trajectory:\t%s'%len(tdf))
print('Points of stops (5min threshold):\t\t\t%s'%len(stdf1))
print('Points of stops (20min threshold):\t\t\t%s'%len(stdf2))
# df = pd.DataFrame(sdf)
# df.to_csv('processed3.csv', index=False)