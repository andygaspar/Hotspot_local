import copy

import numpy as np

from Auction.Agents.ff import FFAgent
from Auction.auction import Auction
from ModelStructure.modelStructure import ModelStructure
from ScheduleMaker import df_to_schedule

schedule_maker = df_to_schedule.RealisticSchedule()


n_flights = 15
c_reduction = 0.5

slot_list, fl_list, airport = schedule_maker.make_sl_fl_from_data(n_flights=n_flights,
                                                                          capacity_reduction=c_reduction,
                                                                          compute=True)

model = ModelStructure(slot_list, fl_list)
A = model.airlines[0]


f = A.flights[0]

f.costFun
approx = lambda m1, j1, m2, j2, t: 0 if t < m1 else (j1 if t < m2 else j2)

from scipy.optimize import curve_fit, minimize

obj = lambda params, times: sum([np.abs(f.costFun(t) - approx(*params, t)) for t in times])


x = np.linspace(0,10,11)
params = (1,2,3,4)
obj(params,x)

solution = minimize(obj, (params), args=(x, approx),
                    method='Powell', options={'maxiter': 10000, 'xtol': 0.5, 'ftol': 0.01})

import pandas as pd

df = pd.read_csv("ScenarioAnalysis/Pax/pax.csv")



