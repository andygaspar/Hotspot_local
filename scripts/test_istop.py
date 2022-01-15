import copy
import time

import pandas as pd

import numpy as np
import random

from GlobalOptimum.globalOptimum import GlobalOptimum
from NNBound.nnBound import NNBoundModel
from ScheduleMaker import df_to_schedule
import udpp_tests.make_sol_df as sol
from UDPP.udppModel import UDPPmodel
from Istop.istop import Istop

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

random.seed(10)
np.random.seed(10)

schedule_maker = df_to_schedule.RealisticSchedule()

n_flights = 120
c_reduction = 0.5

# n_flights = 50

#13.26097559928894  Number of possible offers:  2019

for i in range(10):

    print(i, "++****************************************************")
    slot_list, fl_list, airport = schedule_maker.make_sl_fl_from_data(n_flights=n_flights,
                                                                      capacity_reduction=c_reduction,
                                                                      compute=True)
    if i == i:
        print(airport)
        udpp_model = UDPPmodel(slot_list, fl_list, hfes=0)
        t = time.time()
        udpp_model.run(optimised=True)
        # udpp_model.print_performance()


        fl_list = udpp_model.get_new_flight_list()

        print("istop")
        istop = Istop(slot_list, fl_list, triples=True)
        istop.run(timing=True, verbose=False, branching=True)
        # istop.print_performance()
