import copy
import itertools
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





class BBool(BB):

    def __init__(self, offers, reductions, flights: List[IstopFlight], min_lp_len=80, max_lp_time=10, print_info=100):
        super().__init__(offers, reductions, flights, min_lp_len, max_lp_time, print_info)
        self.precomputed_len = 0
        self.max_precomputed = 0
        self.compatibilityMatrix = np.full((self.numOffers, self.numOffers), False, dtype=bool)
        for i, offer in enumerate(self.offers):
            incompatible = np.unique([off for flight in offer.flights for off in flight.offers])
            indexes = [off.num for off in self.offers if off.num not in incompatible]
            self.compatibilityMatrix[i, indexes] = True


    def run(self):
        self.step(np.full(self.numOffers, False), np.full(self.numOffers, True), 0, self.reductions, self.compatibilityMatrix)

        if len(self.solution) > 0:
            self.solution = [self.offers[i].offer for i in range(self.numOffers) if self.solution[i]]

    def step(self, solution: np.array, offers: np.array, reduction: float, reductions, comp_matrix):

        if self.nodes % self.info == 0:
            print(self.nodes, len(self.precomputed), self.stored, self.precomputed_len, self.max_precomputed)

        self.nodes += 1
        if np.sum(offers) == 0:
            self.initSolution = True
            return 0

        idx = np.nonzero(offers)[0][0]

        l_reduction = reduction + reductions[idx]
        l_solution = copy.copy(solution)
        l_solution[idx] = True

        if l_reduction > self.best_reduction:
            self.solution = l_solution
            self.best_reduction = l_reduction

        l_offers = comp_matrix[idx] * offers
        l_offers[idx] = False

        l_offers_key = np.nonzero(l_offers)[0].tobytes()

        # print(sum(l_offers), "l offers")

        pruned = False
        if self.initSolution:
            if l_offers_key in self.precomputed.keys():
                if self.precomputed[l_offers_key] + l_reduction < self.best_reduction:
                    self.stored += 1
                    self.precomputed_len = (self.precomputed_len * (self.stored - 1) + len(l_offers))/self.stored
                    if self.max_precomputed < len(l_offers):
                        self.max_precomputed = len(l_offers)
                    best_left = self.precomputed[l_offers_key]
                    pruned = True


            else:
                l_offers_reduction = sum(reductions * l_offers)
                bound = l_reduction + l_offers_reduction

                if bound < self.best_reduction:
                    pruned = True
                    best_left = l_offers_reduction

        if not pruned:

            best_left = self.step(l_solution, l_offers, l_reduction, reductions, comp_matrix)

        r_offers = offers
        r_offers[idx] = False
        r_offers_key = np.nonzero(r_offers)[0].tobytes()

        # print(sum(r_offers), "r offers")

        pruned = False

        if r_offers_key in self.precomputed.keys():
            if self.precomputed[r_offers_key] + reduction < self.best_reduction:
                self.stored += 1
                self.precomputed_len = (self.precomputed_len * (self.stored - 1) + len(r_offers))/self.stored
                if self.max_precomputed < len(r_offers):
                    self.max_precomputed = len(r_offers)
                best_right = self.precomputed[r_offers_key]
                pruned = True
        else:
            r_offers_reduction = sum(reductions * r_offers)
            bound = reduction + r_offers_reduction
            if bound < self.best_reduction:
                pruned = True
                best_right = r_offers_reduction

        if not pruned:
            best_right = self.step(solution, r_offers, reduction, reductions, comp_matrix)

        best = max(best_left + reductions[idx], best_right)
        r_offers[idx] = True
        key = np.nonzero(r_offers)[0].tobytes()
        self.precomputed[key] = best

        return best
