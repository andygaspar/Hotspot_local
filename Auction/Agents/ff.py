import numpy as np
import torch

from Auction.Agents.Networks.ff_net import FFNetwork
from Auction.Agents.Networks.ff_net_imitation import FFNetworkImitation
from Auction.Agents.RL.replayMemory import ReplayMemory
from Auction.Agents.agent import Agent
from ModelStructure.modelStructure import ModelStructure


class FFAgent(Agent):

    def __init__(self, airline, other_airlines, start_training=200, sample_size=200, capacity=10_000,
                 cost_normalisation=1000):
        super().__init__()
        self.airline = airline
        self.other_airlines = other_airlines
        self.AI = True
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.startTraining = start_training
        self.sampleSize = sample_size
        self.capacity = capacity

        self.costNormalisation = cost_normalisation

        # cost vect len + positional encoding ==>  airline flight's enconding
        # owner 0 = B, 1 = C + positional encoding in the schedule = => other airlines flight's encoding
        position = 15
        costs = 15
        airline_ = 2
        n_other_airlines = 10
        input_dimension = (self.airline.numFlights + 1) * (costs + position) + (airline_ + position) * n_other_airlines
        self.network = FFNetworkImitation(input_dimension, self.airline.numFlights, 15, self.device)

        self.replayMemory = ReplayMemory(sample_size=self.sampleSize, capacity=self.capacity)

    def set_bids(self, model: ModelStructure, airline, training):

        schedule = np.array([])
        for flight in airline.flights:
            position = np.zeros(model.numFlights)
            position[flight.slot.index] = 1
            schedule = np.concatenate((schedule, flight.costVect/self.costNormalisation, position))

        other_airlines = [air for air in model.airlines if air != airline]
        for air in other_airlines:
            for flight in air.flights:
                other_air_flight = np.zeros(17)
                if air == "B":
                    other_air_flight[0] = 1
                else:
                    other_air_flight[1] = 1
                position = np.zeros(model.numFlights)
                position[flight.slot.index] = 1
                other_air_flight[2:] = position

                schedule = np.concatenate((schedule, other_air_flight))

        input_vects = []
        for flight in airline.flights:
            position = np.zeros(model.numFlights)
            position[flight.slot.index] = 1
            input_vects.append(np.concatenate((flight.costVect/self.costNormalisation, position, schedule)))

        input_vects = torch.from_numpy(np.array(input_vects)).to(self.device).type(dtype=torch.float32)
        actions = self.network.get_bids(input_vects)
        sample = actions.cpu().numpy().reshape(airline.numFlights, model.numFlights + 1)

        if training:
            self.state = input_vects
            self.action = actions
        bids_mat = np.multiply(sample[:, :model.numFlights].T, sample[:, -1]).T
        bids_mat = self.flight_sum_to_zero(bids_mat)
        bids_mat = self.credits_standardisation(bids_mat, airline.credits)

        for i, flight in enumerate(airline.flights):
            flight.bids = bids_mat[i]

    def train(self):
        if self.replayMemory.size > self.startTraining:
            batch = self.replayMemory.get_sample()
            self.network.update_weights(batch)

    def add_record(self, reward, action=None):
        action = self.action if action is None else action.to(self.device)
        reward = torch.tensor(reward).to(self.device)
        self.replayMemory.add_record(self.state, action, reward)
