from typing import List
import numpy as np
from Auction.AirlineAndFlight.auction_flight import AuctionFlight
from itertools import permutations


class DefaultAgent:

    def __init__(self):
        pass

    def set_bids(self, flights: List[AuctionFlight], max_credits):
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
        indexes = list(permutations(range(1, 4)))
        sums = []

        print(bids_mat)
