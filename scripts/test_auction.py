import copy

import numpy as np

from Auction.Agents.ff import FFAgent
from Auction.auction import Auction
from ScheduleMaker import df_to_schedule

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

n_flights = 15
c_reduction = 0.5

slot_list, fl_list, airport = schedule_maker.make_sl_fl_from_data(n_flights=n_flights,
                                                                          capacity_reduction=c_reduction,
                                                                          compute=True)

# airlines = ["A", "B", "C"]
# print(np.random.choice(airlines))
# for flight in fl_list:
#     flight.airlineName = np.random.choice(airlines)


fl_to_airline(fl_list)
print([fl.airlineName for fl in fl_list])
auction = Auction(slot_list, fl_list)

auction.airByName["A"].agent = FFAgent(auction.airByName["A"], auction)

auction.run()
auction.print_performance()

for run in range(10):
    slot_list, fl_list, airport = schedule_maker.make_sl_fl_from_data(n_flights=n_flights,
                                                                              capacity_reduction=c_reduction,
                                                                              compute=True)
    # airlines = ["A", "B", "C"]
    #
    # for flight in fl_list:
    #     flight.airlineName = np.random.choice(airlines)
    # print([flight.airlineName for flight in fl_list])
    # print([flight.eta for flight in fl_list])
    fl_to_airline(fl_list)

    auction.reset(slot_list, fl_list)
    auction.run()
    auction.print_performance()

import numpy as np

a = np.array([[3,3,3], [2,1,2]])

b = np.mean(a, axis=1)

a- b.T