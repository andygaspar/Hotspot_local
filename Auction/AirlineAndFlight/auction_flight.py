import numpy as np

from ModelStructure.Flight import flight as fl

class AuctionFlight(fl.Flight):

    def __init__(self, flight: fl.Flight):

        super().__init__(*flight.get_attributes())

        self.bids = None