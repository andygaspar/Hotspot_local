from typing import List

import numpy as np
import pandas as pd
from itertools import combinations

from Auction.Agents.agent import DefaultAgent
from Auction.AirlineAndFlight.auction_flight import AuctionFlight
from Istop.AirlineAndFlight.istopFlight import IstopFlight
from Istop.Preferences import preference
from ModelStructure.Airline import airline as air
import matplotlib.pyplot as plt


class AuctionAirline(air.Airline):

    def __init__(self, airline_name: str, flights: List[AuctionFlight]):

        super().__init__(airline_name, flights)

        self.flights: List[AuctionFlight]

        self.agent = DefaultAgent()


