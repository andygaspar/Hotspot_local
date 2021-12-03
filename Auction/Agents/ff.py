import numpy as np
import torch

from Auction.Agents.Networks.ff_net import FFNetwork
from Auction.Agents.RL.replayMemory import ReplayMemory
from Auction.Agents.agent import Agent
from ModelStructure.modelStructure import ModelStructure


class FFAgent(Agent):

    def __init__(self, airline, other_airlines):
        super().__init__()
        self.AI = True
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.startTraining = 200
        self.sampleSize = 200
        self.capacity = 10_000

        # cost vect len + positional encoding ==>  airline flight's enconding
        # owner 0 = B, 1 = C + positional encoding in the schedule = => other airlines flight's encoding
        input_dimension = len(airline.flights) * (15 + 15) + (2 + 15) * 10
        self.network = FFNetwork(input_dimension, airline.numFlights, 15, self.device)

        self.replayMemory = ReplayMemory(sample_size=self.sampleSize, capacity=self.capacity)

    def set_bids(self, model: ModelStructure, airline, training):

        input_vect = np.array([])
        for flight in airline.flights:
            position = np.zeros(model.numFlights)
            position[flight.slot.index] = 1
            input_vect = np.concatenate((input_vect, flight.costVect, position))

        other_airlines = [air for air in model.airlines if air != airline]
        for air in other_airlines:
            for flight in air.flights:
                other_air_flight = np.zeros(17)
                if air == "B":
                    other_air_flight[0] = 1
                else:
                    other_air_flight[1] = 1

                other_air_flight[2:] = flight.costVect

                input_vect = np.concatenate((input_vect, other_air_flight))

        input_vect = torch.from_numpy(input_vect).to(self.device).type(dtype=torch.float32)
        action = self.network.get_bids(input_vect)
        sample = action.cpu().numpy().reshape(airline.numFlights, model.numFlights + 1)

        if training:
            self.state = input_vect
            self.action = action
        bids_mat = np.multiply(sample[:, :model.numFlights].T, sample[:, -1]).T
        bids_mat = self.flight_sum_to_zero(bids_mat)
        bids_mat = self.credits_standardisation(bids_mat, airline.credits)

        i = 0
        for flight in airline.flights:
            flight.bids = bids_mat[i]
            i += 1

    def train(self):
        if self.replayMemory.size > self.startTraining:
            batch = self.replayMemory.get_sample()
            self.network.update_weights(batch)

    def add_record(self, reward):
        reward = torch.tensor(reward).to(self.device)
        self.replayMemory.add_record(self.state, self.action, reward)