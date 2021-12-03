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


    def run(self, training=True):

        for airline in self.airlines:
            airline.agent.set_bids(self, airline, training)

        assigned_flights = []
        for slot in self.slots:
            winner = None
            max_bid = -100_000
            for flight in self.flights:
                if flight not in assigned_flights and flight.etaSlot <= slot and flight.bids[slot.index] > max_bid:
                    winner = flight
                    max_bid = flight.bids[slot.index]
            assigned_flights.append(winner)

        for i in range(len(assigned_flights)):
            assigned_flights[i].newSlot = self.slots[i]
        solution.make_solution(self)

        for airline in self.airlines:
            airline.agent.add_record(airline.finalCosts)
            airline.agent.train()

    def reset(self, slots, flights: List[AuctionFlight]):

        self.slots = slots

        self.flights = [flight for flight in flights if flight is not None]

        self.set_flight_index()

        self.set_cost_vect()

        self.set_flights_attributes()

        self.initialTotalCosts = self.compute_costs(self.flights, "initial")

        self.scheduleMatrix = self.set_schedule_matrix()

        self.emptySlots = []

        self.solution, self.report = None, None

        self.df = self.make_df()

        for airline in self.airlines:
            air_flights = [flight for flight in flights if flight.airlineName == airline.name]
            airline.numFlights = len(air_flights)

            airline.flights = air_flights
