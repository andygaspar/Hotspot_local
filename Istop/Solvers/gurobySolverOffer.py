import sys
import time
from typing import List

import numpy as np

# from ...ModelStructure.Flight.flight import Flight
from gurobipy import Model, GRB, quicksum, Env

import time

from Istop.AirlineAndFlight.istopFlight import IstopFlight


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

    def __lt__(self, other):
        return self.num < other.num

def stop(model, where):

    if where == GRB.Callback.MIP:
        objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
        objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
        run_time = model.cbGet(GRB.Callback.RUNTIME)

        if run_time > model._time_limit and abs(objbst - objbnd) < 0.05 * abs(objbst):
            print("stop at", run_time)
            model.terminate()


class GurobiSolverOffer:

    def __init__(self, model, offers, reductions, mip_gap):

        self.m = Model('CVRP')
        # self.m.setParam('Method', 2) ###################testare == 2 !!!!!!!!!!!!111c
        self.m.modelSense = GRB.MAXIMIZE
        self.m.setParam('MIPGap', mip_gap)

        self.flights = model.flights
        self.airlines = model.airlines
        self.slots = model.slots

        self.matches = model.matches
        self.emptySlots = model.emptySlots
        self.flights_in_matches = model.flights_in_matches

        self.f_in_matched = model.f_in_matched
        self.get_match_for_flight = model.get_match_for_flight
        self.check_and_set_matches = model.check_and_set_matches

        self.c = None

        order = np.flip(np.argsort(reductions))
        self.offers = [Offer(offers[j], reductions[j], i) for i, j in enumerate(order)]
        self.numOffers = len(self.offers)
        self.reductions = np.array([reductions[i] for i in order])


        self.set_match_for_flight(model.flights)

        self.compatibilityMatrix = None


    def set_variables(self):

        self.c = self.m.addMVar(self.numOffers, vtype=GRB.BINARY)


    def set_priority(self):

        for i, priority in enumerate(self.reductions):
            self.c[i].setAttr("BranchPriority", int(priority))

    def set_constraints(self):

        # self.compatibilityMatrix = np.full((self.numOffers, self.numOffers), True, dtype=bool)
        # for i, offer in enumerate(self.offers):
        #     incompatible = np.unique([off for flight in offer.flights for off in flight.offers])
        #     indexes = [off.num for off in self.offers if off.num not in incompatible]
        #     self.compatibilityMatrix[i, indexes] = False

        self.compatibilityMatrix = np.zeros((self.numOffers, self.numOffers), dtype=bool)

        for i, offer in enumerate(self.offers):
            incompatible = np.unique([off for flight in offer.flights for off in flight.offers])
            self.compatibilityMatrix[i, incompatible] = True
            self.compatibilityMatrix[i, i] = False

        for i in range(self.numOffers):
            self.compatibilityMatrix[i, i] = False

        self.m.addConstr(self.compatibilityMatrix@self.c <= (np.ones(self.numOffers)-self.c)*self.numOffers)

    def set_objective(self):
        self.flights: List[IstopFlight]

        self.m.setObjective(self.c@self.reductions)

    def run(self, timing=False, verbose=False, time_limit=60, branching=False):

        self.m._time_limit = time_limit
        if not verbose:
            self.m.setParam('OutputFlag', 0)

        self.set_variables()
        if branching:
            self.set_priority()
        start = time.time()
        self.set_constraints()
        end = time.time() - start
        if timing:
            print("Constraints setting time ", end)

        self.set_objective()
        self.m.setParam('Cuts', 3)
        start = time.time()
        # self.m.optimize(stop)
        self.m.optimize()
        end = time.time() - start

        if timing:
            print("Solution time ", end)

        status = None
        if self.m.status == 2:
            status = "optimal"
        if self.m.status == 3:
            status = "infeasible"
        print(status)




        # for flight in self.flights:
        #     if flight.eta > flight.newSlot.time:
        #         print("********************** danno *********************************",
        #               flight, flight.eta, flight.newSlot.time)


        # return self.get_sol_array(), self.get_solution_offers()

    def set_match_for_flight(self, flights: List[IstopFlight]):
        for flight in flights:
            for offer in self.offers:
                match = offer.offer
                for couple in match:
                    if flight.slot == couple[0].slot or flight.slot == couple[1].slot:
                        flight.offers.append(offer.num)

    def get_sol_array(self):
        solution = np.zeros((len(self.flights), len(self.slots)))
        for flight in self.flights:
            for slot in self.slots:
                if self.x[flight.index, slot.index].x > 0.5:
                    solution[flight.index, slot.index] = 1
        return solution

    def get_solution_offers(self):
        solution = np.zeros(len(self.matches))
        for i in range(len(self.matches)):
            if self.c[i].x > 0.5:
                solution[i] = 1
                print(self.matches[i])
        return solution