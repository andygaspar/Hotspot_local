import sys
import time
from typing import List

import numpy as np

# from ...ModelStructure.Flight.flight import Flight
from gurobipy import Model, GRB, quicksum, Env

import time

from Istop.AirlineAndFlight.istopFlight import IstopFlight


def stop(model, where):

    if where == GRB.Callback.MIP:
        objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
        objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
        run_time = model.cbGet(GRB.Callback.RUNTIME)

        if run_time > model._time_limit and abs(objbst - objbnd) < 0.05 * abs(objbst):
            print("stop at", run_time)
            model.terminate()


class GurobiSolver:

    def __init__(self, model):

        self.m = Model('CVRP')
        # self.m.setParam('Method', 2) ###################testare == 2 !!!!!!!!!!!!111c
        self.m.modelSense = GRB.MINIMIZE

        self.flights = model.flights
        self.airlines = model.airlines
        self.slots = model.slots

        self.matches = model.matches
        self.emptySlots = model.emptySlots
        self.flights_in_matches = model.flights_in_matches

        self.f_in_matched = model.f_in_matched
        self.get_match_for_flight = model.get_match_for_flight
        self.check_and_set_matches = model.check_and_set_matches

        self.epsilon = sys.float_info.min

        self.x = None
        self.c = None

        self.reductions = model.reductions

    def set_variables(self):

        self.x = self.m.addVars([(i, j) for i in range(len(self.slots)) for j in range(len(self.slots))],
                                vtype=GRB.BINARY)

        self.c = self.m.addVars([i for i in range(len(self.matches))], vtype=GRB.BINARY)

    def set_priority(self):

        for i in range(len(self.slots)):
            for j in range(len(self.slots)):
                self.x[i, j].setAttr("BranchPriority", 0)

        for i, priority in enumerate(self.reductions):
            self.c[i].setAttr("BranchPriority", int(priority))

    def set_constraints(self):

        self.flights: List[IstopFlight]

        for i in self.emptySlots:
            for j in self.slots:
                self.m.addConstr(self.x[i, j] == 0)

        for flight in self.flights:
            if not self.f_in_matched(flight):
                self.m.addConstr(self.x[flight.index, flight.index] == 1)
            else:
                self.m.addConstr(quicksum(self.x[flight.index, j.index] for j in flight.compatibleSlots) == 1)
                # comp_slot_cons_list = [self.x[flight.index, j.index] for j in flight.compatibleSlots]
                # self.m.addSOS(GRB.SOS_TYPE1, comp_slot_cons_list, list(range(len(comp_slot_cons_list))))

        for j in self.slots:
            self.m.addConstr(quicksum(self.x[i.index, j.index] for i in self.slots) <= 1)
            # self.m.addSOS(GRB.SOS_TYPE1, [self.x[i.index, j.index] for i in self.slots], list(range(len(self.slots))))

        for flight in self.flights:
            for j in flight.notCompatibleSlots:
                self.m.addConstr(self.x[flight.index, j.index] == 0)

        for flight in self.flights_in_matches:
            self.m.addConstr(
                quicksum(self.x[flight.index, slot.index]
                       for slot in self.slots if slot != flight.slot) \
                <= quicksum([self.c[j] for j in self.get_match_for_flight(flight)])
            )
            # offers_for_flights = [self.c[j] for j in self.get_match_for_flight(flight)]
            # self.m.addSOS(GRB.SOS_TYPE1, offers_for_flights, list(range(len(offers_for_flights))))

            self.m.addConstr(quicksum([self.c[j] for j in self.get_match_for_flight(flight)]) <= 1)

        k = 0
        for match in self.matches:
            flights = [flight for pair in match for flight in pair]
            self.m.addConstr(quicksum(quicksum(self.x[i.index, j.index] for i in pair for j in flights)
                                        for pair in match) >= (self.c[k]) * len(flights))

            for pair in match:
                self.m.addConstr(
                    quicksum(self.x[i.index, j.index] * i.fitCostVect[j.index] for i in pair for j in
                           flights) -
                    (1 - self.c[k]) * 10000000 \
                    <= quicksum(self.x[i.index, j.index] * i.fitCostVect[i.index] for i in pair for j in
                              flights) - \
                    self.epsilon)

            k += 1

    def set_objective(self):
        self.flights: List[IstopFlight]

        self.m.setObjective(
            quicksum(self.x[flight.index, j.index] * flight.fitCostVect[j.index]
                   for flight in self.flights for j in self.slots))  # s

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
        # print("problem status, explained: ", self.m.getProbStatusString(), self.m.getObjVal())
        #print(self.m.getObjVal())




        # for flight in self.flights:
        #     if flight.eta > flight.newSlot.time:
        #         print("********************** danno *********************************",
        #               flight, flight.eta, flight.newSlot.time)


        return self.get_sol_array(), self.get_solution_offers()


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