
import preprocessing as pp
import numpy as np
import matplotlib.pyplot as plt

all_frame_features1 = pp.dataprep('data/not_fall/parsed (20).json')

len1 = len(all_frame_features1)

frames1 = np.arange(0,len1)
vel1 = [0]*len1
for i in range(0,len1):
    vel1[i] = all_frame_features1[i][3]
plt.plot(frames1, vel1)
plt.xlabel('Frames')
plt.ylabel('Velocity')
plt.show()
print()

