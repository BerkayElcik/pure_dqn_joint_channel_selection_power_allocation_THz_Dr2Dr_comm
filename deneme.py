
import pandas as pd
import os
import numpy as np

path0 = r"data/LBLRTM_H1_0.1_H2_0.1_ZANGLE_90_RANGE_km_0.001_Season_6_data.csv"

print(path0)
path0 = path0.replace('\\', '\\\\')

print(path0)


transmittance0 = pd.read_csv(path0, header=None)


transmittance0 = transmittance0.set_axis(['vapor', 'transmittance'], axis=1)


print(transmittance0.head())

"""
chkpt_dir='models'
checkpoint_file = os.path.join(chkpt_dir, "deneme")
transmittance0.to_csv(checkpoint_file)
"""

a=15
print(a)
a=np.clip(
                a + 350, 0, 18
            )

print(a)

k=np.zeros(5)
k[3]=1
print(k)


print(np.ones([5,3]))

array=np.array([5,3,2])
array2=5
print(array)
print("egads")

def bin_array(array, m=None):
    # written the code like this in case I want to return back to the version where the agent adds or removes severel channels in a single action
    if m is None:
        m = 15

    if not isinstance(array, np.ndarray):
        array = np.array([array])



    changed_channel = np.zeros(m)
    for disc in array:
        if disc != -1:
            changed_channel[disc] = 1
    return changed_channel

array=bin_array(array, m=30)

print("asg")
print(array)

array2=bin_array(array2, m=30)
print(array2)

print(type(4))