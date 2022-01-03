import sys
from typing import List

import numpy as np

from Istop.AirlineAndFlight.istopFlight import IstopFlight
from gurobipy import Model, GRB, quicksum, Env


class Offer:

    def __init__(self, offer, reduction, num):
        self.offer = offer
        self.reduction = reduction
        self.flights = [flight for couple in offer for flight in couple]
        self.num = num

    def __repr__(self):
        return str(self.num) + " " + ' '.join([f.name for f in self.flights])

    def __eq__(self, other):
        return self.num == other.num


def get_offers_for_flight(flight, r_offers):
    j = 0
    indexes = []
    for offer in r_offers:
        match = offer.offer
        for couple in match:
            if flight.slot == couple[0].slot or flight.slot == couple[1].slot:
                indexes.append(j)
        j += 1
    return indexes


class BB:

    def __init__(self, offers, reductions, flights: List[IstopFlight]):

        self.tree = None
        self.best_reduction = 0
        self.best_bound = 0

        order = np.flip(np.argsort(reductions))
        self.offers = [Offer(offers[j], reductions[j], i) for i, j in enumerate(order)]

        self.set_match_for_flight(flights)
        self.solution = []

        self.nodes = 0
        self.pruned = 0
        self.pruned_lp = 0
        self.initSolution = False

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

        if self.nodes % 10 == 0:
            print("offers", len(offers), "nodes", self.nodes, "pruned", self.pruned, "pruned_lp", self.pruned_lp,
                  "reduction", self.best_reduction)
        if len(offers) == 0:
            self.initSolution = True
            return
        else:
            self.nodes += 1

        l_reduction = reduction + offers[0].reduction
        l_solution = solution + [offers[0]]

        if l_reduction > self.best_reduction:
            self.solution = l_solution
            self.best_reduction = l_reduction

        l_incompatible = [offer for flight in offers[0].flights for offer in flight.offers]
        l_offers = [offer for offer in offers[1:] if offer not in l_incompatible]

        l_lp_bound = self.run_lp(l_offers)

        # print(l_lp_bound + reduction, self.best_reduction)

        if self.initSolution and (reduction + sum([offer.reduction for offer in l_offers]) < self.best_reduction or\
                reduction + l_lp_bound < self.best_reduction):
            self.pruned += 1
            self.pruned_lp += 1
            print("l pruned")
            return

        self.step(l_solution, l_offers, l_reduction)

        r_offers = offers[1:]

        r_lp_bound = self.run_lp(r_offers)

        if reduction + sum([offer.reduction for offer in r_offers]) < self.best_reduction:
            self.pruned += 1
            return

        if reduction + r_lp_bound < self.best_reduction:
            self.pruned += 1
            self.pruned_lp += 1
            print("r pruned")
            return

        self.step(solution, r_offers, reduction)

    def run_lp(self, r_offers):

        flights = []
        for offer in r_offers:
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

        var_type = GRB.BINARY if len(r_offers) < 60 else GRB.CONTINUOUS

        x = m.addVars([(i, j) for i in range(len(flights)) for j in range(len(flights))], vtype=GRB.BINARY, lb=0,
                      ub=1)
        c = m.addVars([i for i in range(len(r_offers))], vtype=var_type, lb=0, ub=1)

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
                       for flight in flights  if flight.airlineName == airline for slot in slots)
            )

        for flight in flights:
            m.addConstr(
                quicksum(x[flight_index[flight], slot_index[slot]]
                       for slot in slots if slot != flight.slot) \
                <= quicksum([c[j] for j in get_offers_for_flight(flight, r_offers)])
            )

            m.addConstr(quicksum([c[j] for j in get_offers_for_flight(flight, r_offers)]) <= 1)

        epsilon = sys.float_info.min

        for k, offer in enumerate(r_offers):
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

        m.optimize()

        initial_cost = sum([flight.cost_fun(flight.slot) for flight in flights])
        final_cost = m.getObjective().getValue()
        return initial_cost - final_cost
