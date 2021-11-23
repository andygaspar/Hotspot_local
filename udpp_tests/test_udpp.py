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

# ciao

schedule_maker = df_to_schedule.RealisticSchedule()


def run_test(n_flights, c_reduction, df, n_runs, test=False,
             mincost=True, nnbound=True, udpp=True, run_num=None, c_red=None, n_f=None):
    test_concluded = False
    i = 0
    print(n_flights, c_reduction)
    while i < n_runs and not test_concluded:

        # print(i)
        # ************* init

        # print(airport)

        # 150 0.2 13
        compute = True
        if test:
            compute = False
            if n_flights == n_f and c_red - 0.01 < c_reduction < c_red + 0.01 and i == run_num:
                compute = True

        slot_list, fl_list, airport = schedule_maker.make_sl_fl_from_data(n_flights=n_flights,
                                                                          capacity_reduction=c_reduction,
                                                                          compute=True)

        mod = Istop(slot_list, fl_list)
        mod.run()
        if compute:
            print(airport)
            # print("i", n_flights, c_reduction, i)
            # istop = Istop(slot_list, fl_list)
            # t = time.time()
            # istop.run(timing=True, verbose = True)
            if mincost:
                print("g", n_flights, c_reduction, i)
                global_model = GlobalOptimum(slot_list, fl_list)
                global_model.run()
                sol.append_to_df(global_model, "mincost")
                # global_model.print_performance()

            if nnbound:
                print("n", n_flights, c_reduction, i)
                max_model = NNBoundModel(slot_list, fl_list)
                max_model.run(verbose=False, time_limit=120)
                sol.append_to_df(max_model, "nnbound")
                # max_model.print_performance()
                #
                # max_model = NNBoundModel(slot_list, fl_list)
                # max_model.run(verbose=True, time_limit=120, rescaling=False)
                # append_to_df(max_model, "nnbound")
                # max_model.print_performance()

            if udpp:
                print("u", n_flights, c_reduction, i)
                udpp_model_xp = UDPPmodel(slot_list, fl_list, hfes=0)
                t = time.time()
                udpp_model_xp.run(optimised=True)
                sol.append_to_df(udpp_model_xp, "udpp_0")
                # print(time.time()-t)


            if not test:
                df = sol.append_results(df, global_model, max_model, udpp_model_xp, i, n_flights, c_reduction, airport,
                                    False)
                print("shape", df.shape, df.memory_usage(deep=True).sum())

            else:
                test_concluded = True
                break

        i += 1

    return df


# in case remember both
random.seed(0)
np.random.seed(0)
# scheduleType = schedule_types(show=True)


df_test = pd.DataFrame(
    columns=["airline", "num flights", "initial costs", "final costs", "reduction %", "run", "n_flights", "c_reduction",
             "model"])

# df_test = pd.concat([df_test, run_test(150, 0.1, df_test, 100)])
for num_flights in [50]:
    for cap_reduction in [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
        df_test = run_test(num_flights, cap_reduction, df_test, 100,
                           test=False, mincost=True, nnbound=True, udpp=True, run_num=9, n_f=150, c_red=0.3)

# test=True, mincost=True, nnbound=True, udpp=True, run_num=9, n_f=150, c_red=0.3
# run_test(num_flights, cap_reduction, 3)


# print(df_test)
df_test.to_csv("udpp_tests/cap_n_fl_test.csv", index_label=False, index=False)

# udpp_model_xp_5 = UDPPmodel(slot_list, fl_list, hfes=5)
# udpp_model_xp_5.run(optimised=True)
# udpp_model_xp_5.report["run"] = i
# udpp_model_xp_5.report["model"] = "udpp_5"
# udpp_model_xp_5.report["comp time"] = [udpp_model_xp_5.computationalTime] + \
#                                     [airline.udppComputationalTime for airline in udpp_model_xp_5.airlines]
#
# protections = [airline.protections for airline in udpp_model_xp_5.airlines]
# udpp_model_xp_5.report["protections"] = [sum(protections)] + protections
#
# positive = [airline.positiveImpact for airline in udpp_model_xp_5.airlines]
# positiveMins = [airline.positiveImpactMinutes for airline in udpp_model_xp_5.airlines]
#
# udpp_model_xp_5.report["positive"] = [sum(positive)] + positive
# udpp_model_xp_5.report["positive mins"] = [sum(positiveMins)] + positiveMins
#
# negative = [airline.negativeImpact for airline in udpp_model_xp_5.airlines]
# negativeMins = [airline.negativeImpactMinutes for airline in udpp_model_xp_5.airlines]
#
# udpp_model_xp_5.report["negative"] = [sum(negative)] + negative
# udpp_model_xp_5.report["negative mins"] = [sum(negativeMins)] + negativeMins
#
# udpp_model_xp_5.report["airport"] = airport
#
#
# udpp_model_xp_5.print_performance()
# print("\n\n\n\n")
