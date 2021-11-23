from typing import List

import numpy as np

from Auction.AirlineAndFlight.auction_airline import AuctionAirline
from Auction.AirlineAndFlight.auction_flight import AuctionFlight
from ModelStructure import modelStructure as ms
from ModelStructure.Flight.flight import Flight
from ModelStructure.Slot.slot import Slot
from ModelStructure.Solution import solution



class Auction(ms.ModelStructure):

    def __init__(self, slot_list: List[Slot], flight_list: List[Flight]):

        flights = [AuctionFlight(flight) for flight in flight_list]

        super().__init__(slot_list, flights, air_ctor=AuctionAirline)

        self.airlines: List[AuctionAirline]

        for airline in self.airlines:
            airline.agent.set_bids(airline.flights, 50)

    def run(self):
        bids_mat = np.array([flight.bids for flight in self.flights])
        print(bids_mat)
        print(np.argmax(bids_mat, axis=1))
        assigned_flights = []
        for slot in self.slots:
            winner = None
            max_bid = -100_000
            for flight in self.flights:
                if flight not in assigned_flights and flight.etaSlot <= slot and flight.bids[slot.index] > max_bid:
                    winner = flight
                    max_bid = flight.bids[slot.index]
            assigned_flights.append(winner)

        print(assigned_flights)
        for i in range(len(assigned_flights)):
            assigned_flights[i].newSlot = self.slots[i]
        solution.make_solution(self)