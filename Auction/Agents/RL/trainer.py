from typing import List

from Auction.AirlineAndFlight.auction_airline import AuctionAirline


class AuctionTrainer:

    def __init__(self, airlines: List[AuctionAirline]):
        self.airlines = airlines

    def set_bids(self, model, training):
        for airline in self.airlines:
            airline.agent.set_bids(model, airline, training)

    def train(self, action=None):
        for airline in self.airlines:
            airline.agent.add_record(airline.finalCosts, action)
            airline.agent.train()
