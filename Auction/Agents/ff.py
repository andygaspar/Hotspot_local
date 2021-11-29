import numpy as np

from Auction.Agents.Networks.ff_net import FFNetwork
from Auction.Agents.agent import Agent
from ModelStructure.modelStructure import ModelStructure


class FFAgent(Agent):

    def __init__(self, airline, other_airlines):
        super().__init__()
        # cost vect len + positional encoding ==>  airline flight's enconding
        # owner 0 = B, 1 = C + positional encoding in the schedule = => other airlines flight's encoding
        input_dimension = len(airline.flights) * (15 + 15) + (2 + 15) * 10
        self.network = FFNetwork(input_dimension, airline.numFlights, 15)

    def set_bids(self, model: ModelStructure, airline):

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

        bids_mat = self.network.get_bids(input_vect)

        bids_mat = self.flight_sum_to_zero(bids_mat)
        bids_mat = self.credits_standardisation(bids_mat, airline.credits)

        i = 0
        for flight in airline.flights:
            flight.bids = bids_mat[i]
            i += 1