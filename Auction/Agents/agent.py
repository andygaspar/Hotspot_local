from abc import ABC, abstractmethod
from itertools import combinations

import numpy as np


class Agent(ABC):

    def __init__(self):
        self.replayMemory = None
        self.AI = False
        self.state = None
        self.action = None
        self.reward = None

    def set_bids(self, model, airline, training):
        pass

    def add_record(self, reward):
        if self.AI:
            self.replayMemory.add_record(self.state, self.action, reward)

    def train(self):
        pass

    @staticmethod
    def flight_sum_to_zero(bids_mat):
        return (bids_mat.T - np.mean(bids_mat, axis=1)).T

    @staticmethod
    def credits_standardisation(bids_mat: np.array, n_credits: int):
        num_flights, bids_size = bids_mat.shape
        indexes = list(combinations(range(bids_size), num_flights))
        sums = [sum([bids_mat[j, indexes[i][j]] for j in range(num_flights)]) for i in range(len(indexes))]
        max_sum = max(sums)
        bids_mat = bids_mat * n_credits / max_sum
        return bids_mat

