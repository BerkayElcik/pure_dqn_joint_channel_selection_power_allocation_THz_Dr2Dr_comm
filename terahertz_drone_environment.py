import numpy as np

import gymnasium as gym
from gymnasium import spaces
from bisect import bisect_left
import pandas as pd

import math
from setuptools import setup



#from gymnasium.envs.registration import register








class thz_drone_env(gym.Env):
    #metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, n_channels=3000, P_T=1, freq_of_movement=0.1, no_of_actions=70):

        """
        path0 = r"data_LBLRTM/LBLRTM_H1_0.1_H2_0.1_ZANGLE_90_RANGE_km_0.001_Season_6_data.csv"
        path0 = path0.replace('\\', '\\\\')

        path1 = r"data_LBLRTM/LBLRTM_H1_0.1_H2_0.1_ZANGLE_90_RANGE_km_0.011_Season_6_data.csv"
        path1 = path1.replace('\\', '\\\\')

        path2 = r"data_LBLRTM/LBLRTM_H1_0.1_H2_0.1_ZANGLE_90_RANGE_km_0.021_Season_6_data.csv"
        path2 = path2.replace('\\', '\\\\')

        path3 = r"data_LBLRTM/LBLRTM_H1_0.1_H2_0.1_ZANGLE_90_RANGE_km_0.031_Season_6_data.csv"
        path3 = path3.replace('\\', '\\\\')

        path4 = r"data_LBLRTM/LBLRTM_H1_0.1_H2_0.1_ZANGLE_90_RANGE_km_0.041_Season_6_data.csv"
        path4 = path4.replace('\\', '\\\\')

        path5 = r"data_LBLRTM/LBLRTM_H1_0.1_H2_0.1_ZANGLE_90_RANGE_km_0.051_Season_6_data.csv"
        path5 = path5.replace('\\', '\\\\')

        path6 = r"data_LBLRTM/LBLRTM_H1_0.1_H2_0.1_ZANGLE_90_RANGE_km_0.061_Season_6_data.csv"
        path6 = path6.replace('\\', '\\\\')

        path7 = r"data_LBLRTM/LBLRTM_H1_0.1_H2_0.1_ZANGLE_90_RANGE_km_0.071_Season_6_data.csv"
        path7 = path7.replace('\\', '\\\\')

        path8 = r"data_LBLRTM/LBLRTM_H1_0.1_H2_0.1_ZANGLE_90_RANGE_km_0.081_Season_6_data.csv"
        path8 = path8.replace('\\', '\\\\')

        path9 = r"data_LBLRTM/LBLRTM_H1_0.1_H2_0.1_ZANGLE_90_RANGE_km_0.091_Season_6_data.csv"
        path9 = path9.replace('\\', '\\\\')

        path10 = r"data_LBLRTM/LBLRTM_H1_0.1_H2_0.1_ZANGLE_90_RANGE_km_0.101_Season_6_data.csv"
        path10 = path10.replace('\\', '\\\\')
        """

        path_freq="data_ITU/freqs_0.5_1.csv"
        path_loss="data_ITU/loss_matrix_0.5_1.csv"
        path_noise="data_ITU/noise_matrix_0.5_1.csv"


        freq_pd=pd.read_csv(path_freq,header=None)
        loss_pd=pd.read_csv(path_loss,header=None)
        noise_pd=pd.read_csv(path_noise,header=None)

        self.freqs_array=freq_pd.to_numpy()
        self.loss_array=loss_pd.to_numpy()
        self.noise_array=noise_pd.to_numpy()




        self.n_channels = n_channels
        self.P_T=P_T
        self.freq_of_movement=freq_of_movement
        self.no_of_actions=no_of_actions

        """
        self.transmittance0 = pd.read_csv(path0, header=None)
        self.transmittance0 = self.transmittance0.set_axis(['vapor', 'transmittance'], axis=1)
        self.transmittance1 = pd.read_csv(path1, header=None)
        self.transmittance1 = self.transmittance1.set_axis(['vapor', 'transmittance'], axis=1)
        self.transmittance2 = pd.read_csv(path2, header=None)
        self.transmittance2 = self.transmittance2.set_axis(['vapor', 'transmittance'], axis=1)
        self.transmittance3 = pd.read_csv(path3, header=None)
        self.transmittance3 = self.transmittance3.set_axis(['vapor', 'transmittance'], axis=1)
        self.transmittance4 = pd.read_csv(path4, header=None)
        self.transmittance4 = self.transmittance4.set_axis(['vapor', 'transmittance'], axis=1)
        self.transmittance5 = pd.read_csv(path5, header=None)
        self.transmittance5 = self.transmittance5.set_axis(['vapor', 'transmittance'], axis=1)
        self.transmittance6 = pd.read_csv(path6, header=None)
        self.transmittance6 = self.transmittance6.set_axis(['vapor', 'transmittance'], axis=1)
        self.transmittance7 = pd.read_csv(path7, header=None)
        self.transmittance7 = self.transmittance7.set_axis(['vapor', 'transmittance'], axis=1)
        self.transmittance8 = pd.read_csv(path8, header=None)
        self.transmittance8 = self.transmittance8.set_axis(['vapor', 'transmittance'], axis=1)
        self.transmittance9 = pd.read_csv(path9, header=None)
        self.transmittance9 = self.transmittance9.set_axis(['vapor', 'transmittance'], axis=1)
        self.transmittance10 = pd.read_csv(path10, header=None)
        self.transmittance10 = self.transmittance10.set_axis(['vapor', 'transmittance'], axis=1)
        """




        self.observation_space = spaces.Dict(
            {
                "channels": spaces.MultiBinary(self.n_channels),
                #"distance": spaces.Discrete(11),# 0.001 km, 0.011 km, 0.021 km, 0.031 km, 0.041 km, 0.051 km, 0.061 km, 0.071 km, 0.081 km, 0.091 km, 0.101 km
                "distance": spaces.Discrete(11, start=1, dtype=np.int32),
                "loss": spaces.Box(low=0,high=1e18, shape=(self.n_channels,), dtype=np.float32),
                "noise": spaces.Box(low=1e-14, high=1e-12, shape=(self.n_channels,), dtype=np.float32),

            }
        )
        # n_channels(0.75 THz - 4.4 THz) as center frequencies for 0.3 GHz wide boxes
        # "transmittance": spaces.Box(0, 1, shape=(self.n_channels,), dtype=np.float32),
        # "capacity": spaces.Box(10e-4,10e4, dtype=np.int64)

        """
        self.observation_space = spaces.Dict(
            {
                "power": spaces.Dict(
                    {
                        "channel_0": spaces.Box(0, 30, shape=(self.n_channels,)),
                    }
                )
                "distance": spaces.Box(0, 100, dtype=int),

            }
        )
        """

        """
        self.action_space = spaces.Dict(
            {
                "channels": spaces.MultiBinary(self.n_channels),
                #"power": spaces.Box(0, self.P_T, shape=(self.n_channels,), dtype=int),
                # n_channels(0.8 THz - 4.3 THz) as center frequencies for 0.1 THz wide boxes
                # 0 dBm corresoponds to 1mW of power, not 0, but I guess it can be considered 0 compared to 30 dBm which corresponds to 10^3 mW
            }
        )
        """
        #Multiple channel selections for 1 action

        """
        self.action_space = spaces.Dict(
            {
                "add_channel": spaces.Box(-1, self.n_channels-1, shape=(10,), dtype=np.int32),
                "remove_channel": spaces.Box(-1, self.n_channels-1, shape=(10,), dtype=np.int32),

            }
        )
        """

        # Multiple channel selections for 1 action
        """
        action_array = (self.n_channels+1) * np.ones(10)
        self.action_space = spaces.MultiDiscrete([action_array,action_array])
        """


        #Single channel selection for 1 action
        """
        self.action_space = spaces.Dict(
            {
                "add_channel": spaces.Discrete(self.n_channels + 1, start=-1),
                "remove_channel": spaces.Discrete(self.n_channels + 1, start=-1),

            }
        )
        """
        self.action_space = spaces.Discrete(self.no_of_actions, dtype=np.int32)




    def _get_obs(self):
        return {
            "channels": self._channels,
            "distance": self._distance,
            #"transmittance": self._transmittance,
            "loss": self._loss,
            "noise": self._noise,
        }

    def _get_info(self):
        return {
            "no_of_channels": np.sum(self._channels),
            "capacity": self._capacity,
            "max_snr": self._max_snr,
            "SNR_list": self._snr_list
        }
    """
    def pow_30(self, channels_obs=None, power_obs=None):
        if channels_obs is None:
            channels_obs = self._channels
        if power_obs is None:
            power_obs = self._power

        # Ensure power_obs is zero where channels_obs is zero
        power_obs = np.where(channels_obs == 0, 0, power_obs)

        # Ensure that the sum of self._power equals 30
        power_sum = np.sum(power_obs)
        if power_sum != 0:  # Avoid division by zero
            scaling_factor = 30 / power_sum
            power_obs = np.round(power_obs * scaling_factor).astype(int)

        # Adjust the sum to exactly 30 if rounding causes a slight discrepancy
        discrepancy = 30 - np.sum(power_obs)
        if discrepancy > 0:
            non_zero_indices = np.where(channels_obs != 0)[0]
            if len(non_zero_indices) > 0:
                power_obs[non_zero_indices[0]] += discrepancy  # Adjust the first non-zero element to fix the sum
        elif discrepancy < 0:
            non_zero_indices = np.where(channels_obs != 0)[0]
            if len(non_zero_indices) < 0:
                power_obs[non_zero_indices[-1]] += discrepancy  # Adjust the last non-zero element to fix the sum

        return power_obs
        
    """


    def EP(self, channels_obs=None, total_power=None):
        if channels_obs is None:
            channels_obs = self._channels
        if total_power is None:
            total_power = self.P_T

        # Ensure it's a numpy array
        channels_obs = np.array(channels_obs)

        #print("channels_obs"+str(channels_obs))
        #print(channels_obs.shape)

        # Number of active channels
        active_channels = np.sum(channels_obs)

        #print("active_channels"+str(active_channels))

        # If no active channels, no power is allocated
        if active_channels == 0:
            return np.zeros_like(channels_obs)

        #print("initial"+str(np.zeros_like(channels_obs)))

        # Initialize power allocation
        power_allocation = np.zeros_like(channels_obs, dtype=np.float32)

        # Equal power distribution to active channels (for simplicity)
        power_per_channel = total_power / active_channels

        #print("power_per_channel"+str(power_per_channel))

        # Allocate power to active channels (where channel_obs == 1)
        power_allocation[channels_obs == 1] = power_per_channel

        return power_allocation



    """
    def take_closest(self, myList, myNumber):
       
        #Assumes myList is sorted. Returns closest value to myNumber.

        #If two numbers are equally close, return the smallest number.
       
        pos = bisect_left(myList, myNumber)
        if pos == 0:
            return myList[0]
        if pos == len(myList):
            return myList[-1]
        before = myList[pos - 1]
        after = myList[pos]
        if after - myNumber < myNumber - before:
            return after
        else:
            return before
    """

    def channel_info(self, distance):
        if distance is None:
            distance = self._distance

        """
        if distance == 0:
            transmittance = self.transmittance0
        elif distance == 1:
            transmittance = self.transmittance1
        elif distance == 2:
            transmittance = self.transmittance2
        elif distance == 3:
            transmittance = self.transmittance3
        elif distance == 4:
            transmittance = self.transmittance4
        elif distance == 5:
            transmittance = self.transmittance5
        elif distance == 6:
            transmittance = self.transmittance6
        elif distance == 7:
            transmittance = self.transmittance7
        elif distance == 8:
            transmittance = self.transmittance8
        elif distance == 9:
            transmittance = self.transmittance9
        elif distance == 10:
            transmittance = self.transmittance10

        transmittance=transmittance["transmittance"].to_numpy()

        transmittance=transmittance[:1217]
        
        return transmittance
        """

        loss = self.loss_array[distance-1]
        noise= self.noise_array[distance-1]




        return loss, noise

    """
    def calc_path_gain(self, freq, tau, distance=None):
        if distance is None:
            distance = self._distance

        c=299792458
        spread_gain=c/(4*math.pi*freq*distance)

        absorption_gain=math.sqrt(tau)

        path_gain=spread_gain*absorption_gain

        return path_gain

    def calc_noise_power(self, tau, T0):

       
        #integral kısmını sor
        

        boltzman_constant=1.38e-23

        emmisivity=1-tau
        T_noise=T0*emmisivity

        noise_power=boltzman_constant*T_noise*0.0003

        return noise_power
        
    """


    def calc_capacity(self, channel_obs, loss, noise):

        if channel_obs is None:
            channel_obs = self._channels
        if loss is None:
            loss=self._loss
        if noise is None:
           noise=self._noise

        power_alloc=self.EP(channel_obs, self.P_T)

        Capacity=0
        SNR_list=[]

        for channel_iter, power_iter in enumerate(power_alloc):

            """
            freq=(channel_iter*0.001)+0.75
            print(freq)
            
            
            tau=transmittance[channel_iter]

            path_gain=self.calc_path_gain(freq, tau, distance)

            temprature=14+273 #average temprature 1 km above sea level in Kelvin

            noise_power=self.calc_noise_power(tau, T0=temprature)
            """

            path_loss=loss[channel_iter]
            noise_power=noise[channel_iter]

            SNR= power_iter/(path_loss*noise_power)

            SNR_list.append(SNR)


            Capacity+=0.1*math.log2(1+SNR)

        max_snr=max(SNR_list)

        return Capacity, max_snr, SNR_list
    """
    def bin_array(self, array, m=None):
        #only input numpy array or a single value, never input list, dict, tuple etc
        #written the code like this in case I want to return back to the version where the agent adds or removes severel channels in a single action
        if m is None:
            m = self.n_channels

        if not isinstance(array, np.ndarray):
            array = np.array([array])

        changed_channel=np.zeros(m)
        for disc in array:
            if disc != -1:
                changed_channel[disc]=1
        return changed_channel

    def bin_list_single_action(self, num, m=None):
        if m is None:
            m = self.n_channels
        changed_channel = np.zeros(m)
        if num != -1:
            changed_channel[num]=1
        return changed_channel
    """



    def action2threshold(self, action):
        """
        if max_snr==None:
            max_snr=self._max_snr
        #won't use max_snr in the current algorithm
        """

        predetermined_starting_value_for_s=20
        start=predetermined_starting_value_for_s
        delta_s=start/(self.no_of_actions-1)
        threshold=action*delta_s
        return threshold

    def threshold2channels(self, channel_obs, threshold, snr_list):
        if channel_obs is None:
            channel_obs = self._channels
        if snr_list is None:
            snr_list=self._snr_list

        channel_snr=np.zeros_like(channel_obs)

        channel_snr[channel_obs == 1] = snr_list

        selected_channels = np.where(channel_snr > threshold, 1, 0)



        return selected_channels





    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        #self._channels=self.np_random.integers(0, 2, size=self.n_channels, dtype=np.int32)
        self._channels=np.ones(shape=(1,self.n_channels), dtype=np.int32)


        #self._power = self.np_random.integers(0, self.P_T, size=self.n_channels, dtype=int)
        #self._power=self.pow_30(self._channels, self._power)



        self._distance=self.np_random.integers(0, 11, dtype=np.int32)

        #self._transmittance = self.channel_info(self._distance)
        self._loss, self._noise = self.channel_info(self._distance)



        """
        print("reset observation")
        print(self._channels)
        print(self._channels.size)
        print(self._distance)
        print(self._distance.size)
        print(self._transmittance)
        print(self._transmittance.size)
        """


        self._capacity, self._max_snr, self._snr_list= self.calc_capacity(self._channels, self._loss, self._noise)


        observation = self._get_obs()
        #print(observation)
        info = self._get_info()
        #print(info)




        #return observation, info
        return observation, info

    def step(self, action):



        """
        if action[0]==-23 and action[1]==-23:
            action=self.action_space.sample()
        """







        #print("action"+str(action))

        #observation=self._get_obs()
        info=self._get_info()

        #self._channels=observation["channels"]

        #self._distance = observation["distance"]

        self._capacity= info["capacity"] #old capacity
        self._max_snr= info["max_snr"]


        threshold = self.action2threshold(action)
        self._channels=self.threshold2channels(self._channels, threshold, self._snr_list)
        self._channels=self._channels.astype(np.int32)





        #self._power=action["power"]
        #self._power = self.pow_30(self._channels, self._power)


        Capacity, self._max_snr, self._snr_list= self.calc_capacity(self._channels, self._loss, self._noise) # new capacity


        reward = Capacity - self._capacity #positive reward if capacity increased, negative reward if capacity decreased





        rng = np.random.random()
        if rng < (self.freq_of_movement / 2):
            self._distance = np.clip(
                self._distance + 1, 0, 10
            )
        elif (self.freq_of_movement / 2) < rng < (self.freq_of_movement / 2):
            self._distance = np.clip(
                self._distance - 1, 0, 10
            )

        self._loss, self._noise = self.channel_info(self._distance)

        self._capacity=Capacity #assign new capacity as the observation


        observation = self._get_obs()
        info = self._get_info()


        #return observation, reward, terminated, False, info

        #might return "truncuated=True" after certain numher of





        return observation, reward, False, False, info
"""
setup(
    name="gym_examples",
    version="0.0.1",
    install_requires=["gymnasium==0.26.0", "pandas", "numpy"],
)
"""