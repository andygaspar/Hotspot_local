from typing import Callable, List, Union

from ModelStructure import modelStructure as mS
from ModelStructure.Airline import airline as air
from ModelStructure.Flight.flight import Flight
from ModelStructure.Solution import solution
from ModelStructure.Slot.slot import Slot
from gurobipy import Model, GRB, quicksum, Env


import numpy as np
import pandas as pd

import time


class GlobalOptimum(mS.ModelStructure):

    def __init__(self, slot_list: List[Slot], flight_list: List[Flight]):

        super().__init__(slot_list, flight_list)

        self.m = Model('CVRP')
        self.m.modelSense = GRB.MINIMIZE
        self.m.setParam('OutputFlag', 0)
        self.x = None

    def set_variables(self):
        flight: Flight
        airline: air.Airline
        self.x = self.m.addVars([(i, j) for i in range(self.numFlights) for j in range(self.numFlights)],
                                vtype=GRB.BINARY)

    def set_constraints(self):

        flight: Flight
        airline: air.Airline
        for flight in self.flights:
            self.m.addConstr(
                quicksum(self.x[flight.index, slot.index] for slot in flight.compatibleSlots) == 1
            )

        for slot in self.slots:
            self.m.addConstr(
                quicksum(self.x[flight.index, slot.index] for flight in self.flights) <= 1
            )

    def set_objective(self):
        flight: Flight
        self.m.setObjective(
            quicksum(self.x[flight.index, slot.index] * flight.cost_fun(slot)
                   for flight in self.flights for slot in self.slots)
        )

    def run(self, timing=False):

        start = time.time()
        self.set_variables()
        self.set_constraints()
        end = time.time() - start
        if timing:
            print("Variables and constraints setting time ", end)

        self.set_objective()

        start = time.time()
        self.m.optimize()
        end = time.time() - start
        if timing:
            print("Simplex time ", end)

        self.assign_flights(self.x)
        self.update_missed_connecting()
        self.update_hitting_curfew()

        solution.make_solution(self)

        for flight in self.flights:
            if flight.eta > flight.newSlot.time:
                print("********************** negative impact *********************************",
                      flight, flight.eta, flight.newSlot.time)



    def assign_flights(self, sol):
        for flight in self.flights:
            for slot in self.slots:
                if sol[flight.index, slot.index].x > 0.5:
                    flight.newSlot = slot
