import sys
import time
from typing import List

import numpy as np
import matplotlib
from _distutils_hack import override
from matplotlib import pyplot as plt


from Istop.AirlineAndFlight.istopFlight import IstopFlight
from gurobipy import Model, GRB, quicksum, Env
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout

from Istop.Solvers.bb import BB, Offer
from Istop.Solvers import bb
from Istop.old.bb_old import get_offers_for_flight


stop = bb.stop


class BB_new_3(BB):

    def __init__(self, offers, reductions, flights: List[IstopFlight], min_lp_len=80, max_lp_time=10, print_info=100):
        super().__init__(offers, reductions, flights, min_lp_len, max_lp_time, print_info)

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
        self.nodes += 1
        if len(offers) == 0:
            self.initSolution = True
            return

        l_reduction = reduction + offers[0].reduction
        l_solution = solution + [offers[0]]

        if l_reduction > self.best_reduction:
            self.update_sol(l_solution, l_reduction)

        l_incompatible = [offer for flight in offers[0].flights for offer in flight.offers]
        l_offers = [offer for offer in offers[1:] if offer not in l_incompatible]

        pruned = False
        if self.initSolution:
            l_offers_reduction = sum([offer.reduction for offer in l_offers])
            bound = l_reduction + l_offers_reduction
            if bound < self.best_reduction:
                pruned = True

        if not pruned:
            self.step(l_solution, l_offers, l_reduction)

        r_offers = offers[1:]

        pruned = False

        r_offers_reduction = sum([offer.reduction for offer in r_offers])
        bound = reduction + r_offers_reduction
        if bound < self.best_reduction:
            pruned = True

        if not pruned:
            self.step(solution, r_offers, reduction)

    def update_sol(self, solution, reduction):
        self.solution = solution
        self.best_reduction = reduction
