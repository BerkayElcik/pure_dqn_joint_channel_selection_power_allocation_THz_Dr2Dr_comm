
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

path0 = r"data_LBLRTM/LBLRTM_H1_0.1_H2_0.1_ZANGLE_90_RANGE_km_0.001_Season_6_data.csv"

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

print("kkkkkkkkkk")
print(np.random.randint(0,500))




path_freq="data_ITU/freqs_0.75_0.8.csv"
path_loss="data_ITU/loss_matrix_0.75_0.8.csv"
path_noise="data_ITU/noise_matrix_0.75_0.8.csv"

freq_pd=pd.read_csv(path_freq)
loss_pd=pd.read_csv(path_loss)
noise_pd=pd.read_csv(path_noise)

freqs_array=freq_pd.to_numpy()
loss_array=loss_pd.to_numpy()
noise_array=noise_pd.to_numpy()

print(loss_array)
print("aaaaarrrrr")
print(loss_array[1])


print(loss_array.max())
print(loss_array.min())

print(noise_array.max())
print(noise_array.min())

print(int(3e+12))



arr1=np.random.randint(15,size=15)
arr2=np.random.randint(15,size=15)

k=len(arr1)

x=np.arange(k)

print(x)
print(arr1)
print(arr2)

fname="arr1"
figure_file = 'plots/' + fname + '.png'
filename=figure_file


fig=plt.figure()


plt.scatter(x, arr1, color="C0")
plt.savefig(filename)
plt.show()


fname="arr2"
figure_file = 'plots/' + fname + '.png'
filename=figure_file



plt.figure()
plt.scatter(x, arr2, color="C1")
plt.savefig(filename)
plt.show()