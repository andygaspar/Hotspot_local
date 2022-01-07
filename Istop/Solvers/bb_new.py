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

class BB_new(BB):

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
            self.update_sol(l_solution, l_reduction, from_mip=False)

        l_incompatible = [offer for flight in offers[0].flights for offer in flight.offers]
        l_offers = [offer for offer in offers[1:] if offer not in l_incompatible]
        offers_key = ".".join([str(offer.num) for offer in l_offers])

        pruned = False
        if self.initSolution:
            if offers_key in self.precomputed.keys():
                if self.precomputed[offers_key] + reduction < self.best_reduction:
                    pruned = True
            else:
                bound = reduction + sum([offer.reduction for offer in l_offers])
                if bound < self.best_reduction:
                    pruned = True
                elif len(l_offers) <= self.min_lp_len:
                    pruned, bound = self.run_and_check_lp(l_offers, l_reduction, l_solution, "LEFT")

                if pruned:
                    self.precomputed[offers_key] = bound
        if not pruned:
            self.step(l_solution, l_offers, l_reduction)

        r_offers = offers[1:]
        offers_key = ".".join([str(offer.num) for offer in r_offers])
        if offers_key in self.precomputed.keys():
            if self.precomputed[offers_key] + reduction < self.best_reduction:
                pruned = True
        else:
            bound = reduction + sum([offer.reduction for offer in r_offers])
            if bound < self.best_reduction:
                pruned = True
            elif len(r_offers) <= self.min_lp_len:
                pruned, bound = self.run_and_check_lp(r_offers, reduction, solution, "RIGHT")

            if pruned:
                self.precomputed[offers_key] = bound

        if not pruned:
            self.step(solution, r_offers, reduction)

    def run_and_check_lp(self, offers, reduction, solution, side):
        pruned = False
        lp_bound, sol = self.run_lp(offers, reduction, self.best_reduction, side, self.max_time)
        if sol is not None:
            solution += sol
            reduction += lp_bound
            self.update_sol(solution, reduction, from_mip=True)
            pruned = True
        elif reduction + lp_bound < self.best_reduction:
            pruned = True

        return pruned, lp_bound

    def prune(self, side, lp=False, precomputed=False):
        self.nodes += 1
        self.pruned += 1
        if side == "LEFT":
            if not lp:
                self.pruned_l_quick += 1
            else:
                self.pruned_l_lp += 1
        else:
            if not lp:
                self.pruned_r_quick += 1
            else:
                self.pruned_r_lp += 1

    def update_sol(self, solution, reduction, from_mip=False):
        self.solution = solution
        self.best_reduction = reduction
        if from_mip:
            self.nodes += 1


    def run_lp(self, offers_, reduction, best_reduction, side, time_limit):

        t = time.time()
        flights = []
        for offer in offers_:
            match = offer.offer
            for couple in match:
                for fl in couple:
                    if fl not in flights:
                        flights.append(fl)
        slots = [flight.slot for flight in flights]
        slot_index = dict(zip(slots, range(len(slots))))
        flight_index = dict(zip(flights, range(len(flights))))

        m = Model('CVRP')
        m.modelSense = GRB.MINIMIZE
        m.setParam('OutputFlag', 0)

        m.setParam('TimeLimit', time_limit)

        # var_type = GRB.BINARY if len(offers_) <= 60 else GRB.CONTINUOUS
        # var = "binary" if var_type == GRB.BINARY else "continuous"

        x = m.addVars([(i, j) for i in range(len(flights)) for j in range(len(flights))], vtype=GRB.BINARY, lb=0, ub=1)
        c = m.addVars([i for i in range(len(offers_))], vtype=GRB.BINARY, lb=0, ub=1)

        for flight in flights:
            m.addConstr(
                quicksum(x[flight_index[flight], slot_index[slot]] for slot in flight.compatibleSlots if slot in slots)
                == 1
            )

        for slot in slots:
            m.addConstr(
                quicksum(x[flight_index[flight], slot_index[slot]] for flight in flights) <= 1
            )

        airlines = []
        for flight in flights:
            if flight.airlineName not in airlines:
                airlines.append(flight.airlineName)

        for airline in airlines:
            m.addConstr(
                quicksum(flight.cost_fun(flight.slot) for flight in flights if flight.airlineName == airline) >= \
                quicksum(x[flight_index[flight], slot_index[slot]] * flight.cost_fun(slot)
                         for flight in flights if flight.airlineName == airline for slot in slots)
            )

        for flight in flights:
            m.addConstr(
                quicksum(x[flight_index[flight], slot_index[slot]]
                         for slot in slots if slot != flight.slot) \
                <= quicksum([c[j] for j in get_offers_for_flight(flight, offers_)])
            )

            m.addConstr(quicksum([c[j] for j in get_offers_for_flight(flight, offers_)]) <= 1)

        epsilon = sys.float_info.min

        for k, offer in enumerate(offers_):
            match = offer.offer
            fls = [flight for pair in match for flight in pair]
            m.addConstr(quicksum(quicksum(x[flight_index[i], flight_index[j]] for i in pair for j in fls)
                                 for pair in match) >= (c[k]) * len(fls))

            for pair in match:
                m.addConstr(
                    quicksum(x[flight_index[i], flight_index[j]] * i.cost_fun(j.slot) for i in pair for j in
                             flights) -
                    (1 - c[k]) * 10000000 \
                    <= quicksum(x[flight_index[i], flight_index[j]] * i.cost_fun(i.slot) for i in pair for j in
                                flights) - \
                    epsilon)

        m.setObjective(
            quicksum(x[flight_index[flight], slot_index[slot]] * flight.cost_fun(slot)
                     for flight in flights for slot in slots)
        )

        initial_cost = sum([flight.cost_fun(flight.slot) for flight in flights])
        m._initial_cost = initial_cost
        m._reduction = reduction
        m._best_reduction = best_reduction
        m._side = side
        m.optimize(stop)

        branch_reduction = initial_cost - m.ObjBound

        # status = "OPTIMAL" if (m.status == 2 and reduction + branch_reduction > best_reduction) \
        #     else "bound" if m.status != 2 else "OPT_bound"
        #
        # print(status, len(offers_), time.time() - t,
        #       reduction + branch_reduction, best_reduction, side)
        # final_cost = m.getObjective().getValue()

        if m.status == 2 and reduction + branch_reduction > best_reduction:
            solution = []
            for i in range(len(offers_)):
                if c[i].x > 0.5:
                    solution.append(offers_[i])
        else:
            solution = None

        return branch_reduction, solution


