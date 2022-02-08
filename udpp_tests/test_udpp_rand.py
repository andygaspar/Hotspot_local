import time

import pandas as pd

import numpy as np
import random

from GlobalOptimum.globalOptimum import GlobalOptimum
from NNBound.nnBound import NNBoundModel
import udpp_tests.make_sol_df as sol
from ScheduleMaker.real_schedule import RealisticSchedule, Regulation
from UDPP.udppModel import UDPPmodel

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

schedule_maker = RealisticSchedule()


def run_test(df, n_runs, capacity_min: float = 0.,
             n_flights_min: int = 0,
             n_flights_max: int = 1000,
             start: int = 0,
             end: int = 1441):

    for i in range(n_runs):
        regulation: Regulation
        regulation = schedule_maker.get_regulation()
        print(regulation.airport)
        print(regulation.nFlights, regulation.cReduction, regulation.startTime, i)
        slot_list, fl_list = schedule_maker.make_sl_fl_from_data(airport=regulation.airport,
                                                                 n_flights=regulation.nFlights,
                                                                 capacity_reduction=regulation.cReduction,
                                                                 compute=True, regulation_time=regulation.startTime)



        global_model = GlobalOptimum(slot_list, fl_list)
        global_model.run()
        sol.append_to_df(global_model, "mincost")
        # global_model.print_performance()

        # print("n", regulation.nFlights, regulation.cReduction, i)
        max_model = NNBoundModel(slot_list, fl_list)
        max_model.run(verbose=False, time_limit=120, rescaling=False)
        sol.append_to_df(max_model, "nnbound")
        # max_model.print_performance()

        # print("u", regulation.nFlights, regulation.cReduction, i)
        udpp_model = UDPPmodel(slot_list, fl_list, hfes=5)
        t = time.time()
        udpp_model.run(optimised=True)
        # udpp_model.print_performance()
        sol.append_to_df(udpp_model, "udpp", hfes=5)
        # print(time.time()-t)

        df = sol.append_results(df, global_model, max_model, udpp_model, i, regulation.nFlights, regulation.cReduction,
                                regulation.airport, regulation.startTime, True)

    return df


# in case remember both
random.seed(0)
np.random.seed(0)
# scheduleType = schedule_types(show=True)


df_test = pd.DataFrame(
    columns=["airline", "num flights", "initial costs", "final costs", "reduction %", "run", "n_flights", "c_reduction",
             "model"])

df_test = run_test(n_runs=1000, df=df_test)
df_test.to_csv("udpp_tests/cap_n_fl_test_1000_hfes.csv", index_label=False, index=False)