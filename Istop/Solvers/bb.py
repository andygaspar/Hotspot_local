from typing import List

import numpy as np

from Istop.AirlineAndFlight.istopFlight import IstopFlight



class Node:

    def __init__(self):
        self.node = None

class Offer:

    def __init__(self, offer, reduction, num):
        self.offer = offer
        self.reduction = reduction
        self.flights = [flight for couple in offer for flight in couple]
        self.num = num

    def __repr__(self):
        return str(self.num)

    def __eq__(self, other):
        return self.num == other.num


class BB:

    def __init__(self, offers, reductions, flights: List[IstopFlight]):
        self.tree = None
        self.best_reduction = 0

        order = np.flip(np.argsort(reductions))
        self.offers = [Offer(offers[j], reductions[j], i) for i,j in enumerate(order)]

        self.set_match_for_flight(flights)
        self.solution = []

        self.nodes = 0
        self.pruned = 0

    def set_match_for_flight(self, flights: List[IstopFlight]):
        for flight in flights:
            for offer in self.offers:
                match = offer.offer
                for couple in match:
                    if flight.slot == couple[0].slot or flight.slot == couple[1].slot:
                        flight.offers.append(offer)

    def run(self):
        self.step([], self.offers, 0)
        if len(self.solution) > 0:
            self.solution = [offer.offer for offer in self.solution]

    def step(self, solution: List[Offer], offers: list[Offer], reduction: float):

        if self.nodes % 5000 == 0:
            print("nodes", self.nodes, "pruned", self.pruned)
        if len(offers) == 0:
            return
        else:
            self.nodes += 1

        l_reduction = reduction + offers[0].reduction
        l_solution = solution + [offers[0]]

        if l_reduction > self.best_reduction:
            self.solution = l_solution
            self.best_reduction = l_reduction
            print("sol", self.nodes, self.best_reduction, self.solution)

        l_incompatible = [offer for flight in offers[0].flights for offer in flight.offers]
        l_offers = [offer for offer in offers if offer not in l_incompatible]

        self.step(l_solution, l_offers, l_reduction)

        r_offers = offers[1:]

        if reduction + sum([offer.reduction for offer in r_offers]) < self.best_reduction:
            self.pruned += 1
            return

        self.step(solution, r_offers, reduction)
