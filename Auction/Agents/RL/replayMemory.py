from typing import List
import numpy as np
from csv import writer

import pandas as pd
import torch


class ReplayMemory:
    size: int
    states: List[np.array]
    actions: List[int]
    nextStates: List[np.array]
    rewards: List[float]
    dones: List[bool]
    sampleSize: int
    capacity: int

    def __init__(self, sample_size: int, capacity: int):
        self.size = 0
        self.states = []
        self.actions = []
        self.rewards = []
        self.sampleSize = sample_size
        self.capacity = capacity
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def add_record(self, state, action, reward):
        if len(self.actions) >= self.capacity:
            self.states = self.states[1:]
            self.actions = self.actions[1:]
            self.rewards = self.rewards[1:]
            self.size -= 1
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.size += 1

    def get_sample(self):
        random_idx = np.random.choice(range(self.size), size=self.sampleSize, replace=False).astype(int)
        return [self.states[i] for i in random_idx], [self.actions[i] for i in random_idx],\
               [self.rewards[i] for i in random_idx]

    def export_memory(self):
        with open("states.csv", 'a+', newline='') as write_obj:
            for state in self.states:
                csv_writer = writer(write_obj)
                csv_writer.writerow(state.cpu().tolist())
        with open("states.csv", 'a+', newline='') as write_obj:
            npthing = np.loadtxt("states.csv", dtype=np.float32)
            print(npthing)

        with open("actions.csv", 'a+', newline='') as write_obj:
            csv_writer = writer(write_obj)
            csv_writer.writerow(self.actions)

        with open("rewards.csv", 'a+', newline='') as write_obj:
            csv_writer = writer(write_obj)
            csv_writer.writerow(self.rewards)

    def import_memory(self):
        import csv
        states = []
        with open("replay_memory/states.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            for row in csv_reader:
                states.append(torch.tensor(state))



        with open("replay_memory/rewards.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            reward = []
            for row in csv_reader:
                reward += row

        with open("replay_memory/actions.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            actions = []
            for acts in csv_reader:
                actions += torch.tensor(acts).to(self.device)
            actions = np.array(actions).astype(int).tolist()

