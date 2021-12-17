import copy

import numpy as np
import torch

from Auction.Agents.ff import FFAgent
from Auction.auction import Auction
from ScheduleMaker import df_to_schedule
from Auction.Agents.RL.trainer import AuctionTrainer

schedule_maker = df_to_schedule.RealisticSchedule()


def fl_to_airline(flights):
    unassigned = copy.copy(flights)
    for i in range(5):
        f = np.random.choice(unassigned)
        f.airlineName = "A"
        unassigned.remove(f)
    f = np.random.choice(unassigned)
    f.airlineName = "B"
    unassigned.remove(f)
    f = np.random.choice(unassigned)
    f.airlineName = "C"
    unassigned.remove(f)

    for flight in unassigned:
        flight.airlineName = np.random.choice(["B", "C"])


def auction_step(auction, trainer, training, slot_list=None, fl_list=None, initial=False, print_p=False, action=None):
    if not initial:
        auction.reset(slot_list, fl_list)
    trainer.set_bids(auction, training=training)
    auction.run()
    if training:
        trainer.train(action)
    if print_p:
        auction.print_performance()

n_flights = 15
c_reduction = 0.5

slot_list, fl_list, airport = schedule_maker.make_sl_fl_from_data(n_flights=n_flights,
                                                                          capacity_reduction=c_reduction,
                                                                          compute=True)

# airlines = ["A", "B", "C"]
# print(np.random.choice(airlines))
# for flight in fl_list:
#     flight.airlineName = np.random.choice(airlines)

start_training = 250

fl_to_airline(fl_list)
print([fl.airlineName for fl in fl_list])
auction = Auction(slot_list, fl_list)
auction.airByName["A"].agent = FFAgent(auction.airByName["A"], auction, start_training=start_training, sample_size=250)
trainer = AuctionTrainer(auction.airlines)
auction2 = Auction(slot_list, fl_list)
trainer2 = AuctionTrainer(auction2.airlines)


auction_step(auction, trainer, training=False, initial=True, print_p=True)
auction_step(auction2, trainer2, training=False, initial=True, print_p=True)

iteration = 0
for run in range(1):
    for r in range(1):
        iteration = run * 100 + r * 10
        print(iteration)
        for j in range(5):
            slot_list, fl_list, airport = schedule_maker.make_sl_fl_from_data(n_flights=n_flights,
                                                                                      capacity_reduction=c_reduction,
                                                                                      compute=True)

            fl_to_airline(fl_list)

            auction_step(auction2, trainer2, training=True, slot_list=slot_list, fl_list=fl_list)
            action = torch.ones((5,16))
            for i, flight in enumerate(auction2.airByName["A"].flights):
                action[i, :15] = torch.tensor(flight.bids)
            auction_step(auction, trainer, training=True, slot_list=slot_list, fl_list=fl_list, action=action)
        print("loss:", auction.airByName["A"].agent.network.loss)
        print("error", auction.airByName["A"].agent.network.error)

    auction.airByName["A"].agent.network.print_params()

    if run > start_training:
        for test_run in range(5):
            slot_list, fl_list, airport = schedule_maker.make_sl_fl_from_data(n_flights=n_flights,
                                                                              capacity_reduction=c_reduction,
                                                                              compute=True)

            fl_to_airline(fl_list)
            print(" ")
            auction_step(auction, trainer, training=False, slot_list=slot_list, fl_list=fl_list, print_p=True)
            auction_step(auction2, trainer2, training=False, slot_list=slot_list, fl_list=fl_list, print_p=True)


auction.airByName["A"].agent.replayMemory.export_memory()