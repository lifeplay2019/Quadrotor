# python3
# -*- coding: utf-8 -*-
"""The file used to implement the data store and replay

By YuXin Hu
2023.11.20
"""

from collections import deque
import random
import numpy as np

"""
********************************************************************************************************

********************************************************************************************************/
"""


class ReplayBuffer(object):
    """ storing data in order replaying for train algorithm"""

    def __int__(self, buffer_size, random_seed=100):
        # size of min buffer is able to train
        self.buffer_size = buffer_size
        # count for the replay buffer, the buffer contains all the data together
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)
        self.isBufferFull = False
        # counter for episode number
        # episodePos is the position of the start for each buffer in each episode
        # episodeRew is the sun rewards of step for each episode
        self.episodeNum = 0
        self.episodePos = deque()
        self.episodeRew = deque()

    def buffer_append(self, experience):
        """" append data for buffer """
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
            self.isBufferFull = False
        else:
            self.buffer.popleft()
            self.buffer_append(experience)
            self.isBufferFull = True

    def episode_append(self, rewards):
        self.episodeNum += 1
        self.episodePos.append(self.count)
        self.episodeRew.append(rewards)

    def size(self):
        return self.count

    def buffer_sample_batch(self, batch_size):
        """branch of data with a size of branch_size"""
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)
        return batch

    def clear(self):
        """ clear the remaining data"""
        self.buffer.clear()
        self.count = 0
        self.episodeNum = 0
        self.episodePos.clear()
        self.episodeRew.clear()


class DataRecord(object):

    def __init__(self, compatibility_mode=False):
        # new buffer, storagedatain mew episode
        self.episodeList = list()
        self.bufferTemp = deque()
        self.compatibilityMode = compatibility_mode

        # counter for reply buffer and episode
        self.count = 0
        self.episodeNum = 0
        # record new episode record (sum) of the steps
        self.episodeRew = deque()
        # record the average Td error
        self.episodeTdErr = deque()
        # record the weight
        self.episodeWeights = deque()
        # record some weight, once each step, for vary weight
        self.weights = deque()

        if self.compatibilityMode:
            self.buffer = deque()
            self.episodePos = deque()

    def buffer_append(self, experience, weight = 0):
        """append data to buffer, run as each step as the system upgrade"""
        self.bufferTemp.append(experience)
        self.count +=1
        self.weights.append(weights)

        if self.compatibilityMode:
            self.buffer.append(experience)
            ###
            ###

    def episode_append(self, rewards=0, td_err=0, weights=0):
        """append data to episode buffer, should run each episode after episode finish"""
        self.episodeNum += 1
        self.episodeRewards.append(rewards)
        self.episodeTdErr.append(td_err)
        self.episodeWeights.append(weights)

        self.episodeList.append(self.bufferTemp)
        self.bufferTemp = deque()
        if self.compatibilityMode:
            self.episodePos.append(self.count)

    def get_episode_buffer(self, index=-1):
        if index == -1:
            index = self.episodeNum - 1
        elif index > (self.episodeNum - 1):
            self.print_mess("Does not exist this episode!")
        else:
            index = index
            return

        buffer_temp = self.episodeList[index]
        data = list()
        item_len = len(buffer_temp[0])
        for ii in range(item_len):
            x = np.array([_[ii] for _ in buffer_temp])
            data.append(x)
        return data

    def size(self):
        return self.count

    def clear(self):
        self.count = 0
        self.episodeNum = 0
        self.episodeRewards.clear()
        self.bufferTemp.clear()
        self.episodeList.clear()
        if self.compatibilityMode:
            self.buffer.clear()
            self.episodePos.clear()

    @classmethod
    def print_mess(cls, mes=""):
        # implement with print or warning if the project exist
        print(mes)

