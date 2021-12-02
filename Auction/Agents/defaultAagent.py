from typing import List
import numpy as np

from Auction.Agents.agent import Agent
from Auction.AirlineAndFlight.auction_flight import AuctionFlight
from itertools import combinations


class DefaultAgent(Agent):

    def __init__(self):
        super().__init__()

    def set_bids(self, model, airline, training):
        flights = airline.flights
        bids_mat = np.zeros((len(flights), flights[0].costVect.shape[0]))
        j = 0
        for flight in flights:
            bids = np.zeros(flight.costVect.shape[0])
            mean = np.mean(flight.costVect)
            for i in range(flight.etaSlot.index, bids.shape[0]):
                bids[i] = mean - flight.costVect[i]
            bids_mat[j, :] = bids
            j += 1
            flight.bids = bids
        # indexes = list(combinations(range(flights[0].costVect.shape[0]), len(flights)))
        # sums = [sum([bids_mat[j, indexes[i][j]] for j in range(len(flights))]) for i in range(len(indexes))]
        # max_sum = max(sums)
        bids_mat = self.credits_standardisation(bids_mat, airline.credits)

        i = 0

        for flight in flights:
            flight.bids = bids_mat[i]
            i += 1




