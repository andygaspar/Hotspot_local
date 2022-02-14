import copy
import time

import pandas as pd

import numpy as np
import random

from GlobalOptimum.globalOptimum import GlobalOptimum
from NNBound.nnBound import NNBoundModel
from ScheduleMaker import df_to_schedule
import udpp_tests.make_sol_df as sol
from ScheduleMaker.real_schedule import RealisticSchedule
from UDPP.udppModel import UDPPmodel
from Istop.istop import Istop

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

random.seed(10)
np.random.seed(10)

schedule_maker = RealisticSchedule()

n_flights = 120
c_reduction = 0.5
mip_gap = 0.01

# n_flights = 80



for i in range(20):
    regulation = schedule_maker.get_regulation()

    print(i, "++****************************************************")
    slot_list, fl_list = schedule_maker.make_sl_fl_from_data(airport=regulation.airport,
                                                             n_flights=regulation.nFlights,
                                                             capacity_reduction=regulation.cReduction,
                                                             compute=True, regulation_time=regulation.startTime)
    if i == i:
        udpp_model = UDPPmodel(slot_list, fl_list, hfes=0)
        t = time.time()
        udpp_model.run(optimised=True)
        udpp_model.print_performance()


        fl_list = udpp_model.get_new_flight_list()

        print("istop")
        istop = Istop(slot_list, fl_list, triples=True, mip_gap=mip_gap)

        istop.run()
        istop.print_performance()
#