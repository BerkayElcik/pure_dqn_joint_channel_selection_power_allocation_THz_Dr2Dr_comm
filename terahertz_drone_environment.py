import numpy as np

import gymnasium as gym
from gymnasium import spaces
from bisect import bisect_left
import pandas as pd

from math import log2

#from gymnasium.envs.registration import register








class thz_drone_env(gym.Env):
    #metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, n_channels=35, P_T=30):
        self.n_channels = n_channels
        #self.L=L
        self.P_T=P_T

        self.path_loss_10m = pd.read_csv("path_loss_for_10m")
        self.path_loss_50m = pd.read_csv("path_loss_for_50m")
        self.path_loss_100m = pd.read_csv("path_loss_for_100m")

        self.noise_power_10m = pd.read_csv("noise_power_10m")
        self.noise_power_50m = pd.read_csv("noise_power_50m")
        self.noise_power_100m = pd.read_csv("noise_power_100m")

        self.observation_space = spaces.Dict(
            {
                "channels": spaces.MultiBinary(self.n_channels),
                "power": spaces.Box(0, self.P_T, shape=(self.n_channels,), dtype=int),
                # n_channels(0.8 THz - 4.3 THz) as center frequencies for 0.1 THz wide boxes
                # 0 dBm corresoponds to 1mW of power, not 0, but I guess it can be considered 0 compared to 30 dBm which corresponds to 10^3 mW
                "distance": spaces.Discrete(3), # 10m, 50m, 100m
                "path_loss": spaces.Box(110, 200, shape=(self.n_channels), dtype=float), # as dB
                "noise_power": spaces.Box(0, 200, shape=(self.n_channels), dtype=float), # as dB (don't know what values to put)
                "capacity": spaces.Box(10e-4,10e4, dtype=int)
            }
        )


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


        self.action_space = spaces.Dict(
            {
                "channels": spaces.MultiBinary(self.n_channels),
                "power": spaces.Box(0, self.P_T, shape=(self.n_channels,), dtype=int),
                # n_channels(0.8 THz - 4.3 THz) as center frequencies for 0.1 THz wide boxes
                # 0 dBm corresoponds to 1mW of power, not 0, but I guess it can be considered 0 compared to 30 dBm which corresponds to 10^3 mW
            }
        )
        #self.action_space = spaces.Box(0, self.P_T, shape=(self.n_channels,), dtype=int)

    def _get_obs(self):
        return {
            "channels": self._channels,
            "power": self._power,
            "distance": self._distance,
            "path_loss": self._path_loss,
            "noise_power": self._noise_power,
            "capacity": self._capacity
        }

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

    def take_closest(self, myList, myNumber):
        """
        Assumes myList is sorted. Returns closest value to myNumber.

        If two numbers are equally close, return the smallest number.
        """
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

    def channel_info(self, distance):
        if distance is None:
            distance = self._distance

        if distance==0:
            path_loss_pd=self.path_loss_10m
            noise_power_pd=self.noise_power_10m
        elif distance==1:
            path_loss_pd = self.path_loss_50m
            noise_power_pd = self.noise_power_50m
        elif distance==2:
            path_loss_pd = self.path_loss_100m
            noise_power_pd = self.noise_power_100m


        path_loss=path_loss_pd.to_numpy(dtype=float)
        noise_power = noise_power_pd.to_numpy(dtype=float)

        return path_loss, noise_power, path_loss_pd, noise_power_pd

    def calc_capacity(self, path_loss_pd, noise_power_pd, power_obs=None):

        if power_obs is None:
            power_obs = self._power

        Capacity=0
        for channel_iter, power_iter in enumerate(power_obs):
            freq=(channel_iter*0.1)+0.8
            freq_list=path_loss_pd["frequency"].tolist()


            freq=self.take_closest(freq_list, freq)

            path_loss = path_loss_pd[path_loss_pd["frequency"] == freq]["loss"]
            noise_power = noise_power_pd[noise_power_pd["frequency"] == freq]["noise"]


            SNR= (power_iter*path_loss)/noise_power

            Capacity+=0.1*log2(1+SNR)

        return Capacity



    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._channels=self.np_random.integers(0, 2, size=self.n_channels, dtype=int)
        self._power = self.np_random.integers(0, self.P_T, size=self.n_channels, dtype=int)
        self._power=self.pow_30(self._channels, self._power)



        self._distance=self.np_random.integers(0, 3, dtype=int)

        self._path_loss, self._noise_power, path_loss_pd, noise_power_pd = self.channel_info(self._distance)

        self._capacity = self.calc_capacity(self._power, path_loss_pd, noise_power_pd)


        observation = self._get_obs()
        #info = self._get_info()


        #return observation, info
        return observation

    def step(self, action):

        observation=self._get_obs()# if this does not work, take observation as an input to step function


        self._distance = observation["distance"]
        self._path_loss, self._noise_power, path_loss_pd, noise_power_pd = self.channel_info(self._distance)

        self._capacity= observation["capacity"] #old capacity

        self._channels = action["channels"]
        self._power=action["power"]
        self._power = self.pow_30(self._channels, self._power)



        #terminated = np.array_equal(self._agent_location, self._target_location)






        Capacity=self.calc_capacity(self._power, path_loss_pd, noise_power_pd) # new capacity


        reward = self._capacity - Capacity #positive reward if capacity increased, negative reward if capacity decreased

        self._capacity=Capacity #assign new capacity as the observation


        observation = self._get_obs()
        #info = self._get_info()


        #return observation, reward, terminated, False, info

        #might return "truncuated=True" after certain numher of

        return observation, reward, False, False