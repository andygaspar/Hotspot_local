from typing import List

import numpy as np

from Istop.AirlineAndFlight.istopFlight import IstopFlight



class Node:

    def __init__(self):
        self.node = None

class Offer:

    def __init__(self, offer, reduction):
        self.offer = offer
        self.reduction = reduction
        self.flights = [flight for couple in offer for flight in couple]


class BB:

    def __init__(self, offers, reductions, flights: List[IstopFlight]):
        self.tree = None
        self.best_reduction = 0

        order = np.flip(np.argsort(reductions))
        self.offers = [Offer(offers[i], reductions[i]) for i in order]

        self.set_match_for_flight(flights)
        self.solution = []

        self.nodes = 0

    def set_match_for_flight(self, flights: List[IstopFlight]):
        for flight in flights:
            for offer in self.offers:
                match = offer.offer
                for couple in match:
                    if flight.slot == couple[0].slot or flight.slot == couple[1].slot:
                        flight.offers.append(match)

    def step(self, solution: List[Offer], offers: list[Offer], reduction: float):
        self.nodes += 1
        if len(offers) == 0:
            return

        l_reduction = reduction + offers[0].reduction
        l_solution = solution + [offers[0]]

        if l_reduction > self.best_reduction:
            self.solution = l_solution

        l_incompatible = [offer for flight in offers[0].flights for offer in flight.offers]
        l_offers = [offer for offer in offers if offer not in l_incompatible]