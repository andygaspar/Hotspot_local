import numpy as np

from Auction.auction import Auction
from ScheduleMaker import df_to_schedule

schedule_maker = df_to_schedule.RealisticSchedule()
n_flights = 15
c_reduction = 0.5

slot_list, fl_list, airport = schedule_maker.make_sl_fl_from_data(n_flights=n_flights,
                                                                          capacity_reduction=c_reduction,
                                                                          compute=True)
airlines = ["A", "B", "C"]
print(np.random.choice(airlines))
for flight in fl_list:
    flight.airlineName = np.random.choice(airlines)


auction = Auction(slot_list, fl_list)
auction.run()
auction.print_performance()