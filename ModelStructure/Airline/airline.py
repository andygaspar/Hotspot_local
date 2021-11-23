from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Callable
from itertools import combinations
from ModelStructure.Flight.flight import Flight
from ModelStructure.Slot.slot import Slot

low_cost = pd.read_csv("ScenarioAnalysis/df_frequencies/2017-LCC.csv")


class Airline:

    def __init__(self, airline_name: str, flights: List[Flight]):

        self.name = airline_name

        self.lowCost = True if airline_name in low_cost.airline.to_list() else False

        self.numFlights = len(flights)

        self.flights = flights

        self.AUslots = np.array([flight.slot for flight in self.flights])

        self.initialCosts = None

        self.finalCosts = None

        self.protections = 0

        self.udppComputationalTime = 0

        self.positiveImpact = 0

        self.positiveImpactMinutes = 0

        self.negativeImpact = 0

        self.negativeImpactMinutes = 0

        for i in range(len(self.flights)):
            self.flights[i].set_local_num(i)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        if type(other) == str:
            return self.name == other
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)
