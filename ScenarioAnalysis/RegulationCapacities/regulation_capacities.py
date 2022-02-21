import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ScenarioAnalysis.RegulationCapacities.df_refactor import get_airport_set_dicts

pd.set_option('display.max_columns', None)
plt.rcParams["figure.figsize"] = (20,10)

cap = pd.read_csv("ScenarioAnalysis/RegulationCapacities/capacities.csv")
airports = pd.read_csv("ScenarioAnalysis/airportMore25M.csv")
cap = cap[cap.ReferenceLocationName.isin(airports.Airport)]

for airport in airports.Airport:
    if airport not in cap.ReferenceLocationName.unique():
        print(airport)


cap.ReferenceLocationName.unique()
airports.Airport
airport_max_capacity = []
for airport in airports.Airport:
    airport_max_capacity.append(max(cap[cap.ReferenceLocationName == airport].Capacity))

df_airport_max_capacity = pd.DataFrame({"airport": airports.Airport, "capacity": airport_max_capacity})
df_airport_max_capacity.to_csv("ScenarioAnalysis/RegulationCapacities/airport_max_capacity.csv")
set1, set2 = get_airport_set_dicts()

airport_list = airports.Airport.to_list() + list(set1.keys()) + list(set2.keys())


reg = pd.read_csv("ScenarioAnalysis/RegulationCapacities/regulations.csv")
reg = reg[reg.capacity_reduction > 0]


reg_airp = reg[reg.ReferenceLocationName.isin(airports.Airport)]

airport_regulations = reg_airp.value_counts("ReferenceLocationName")

reg_n_flights = reg_airp.drop_duplicates(["RegulationID"])
max_range = max(reg_n_flights.n_flights)//20 * 20 + 20
frequency, bins = np.histogram(reg_n_flights.n_flights, bins=max(reg_n_flights.n_flights)//20 + 1, range=(0, max_range))

plt.bar(range(frequency.shape[0]), frequency)
plt.xticks(range(frequency.shape[0]), [str(i*20)+"-"+str((i+1)*20) for i in range(frequency.shape[0])])
plt.title("number of delayed flights per regulation distribution")
plt.show()



frequency_n_flights = pd.DataFrame({"n_flights": bins[1:].astype(int), "frequency": frequency})
frequency_n_flights.to_csv("ScenarioAnalysis/df_frequencies/n_flights_frequency.csv", index_label=False, index=False)





frequency, bins = np.histogram(reg_n_flights.capacity_reduction, bins=10, range=(0,1))
plt.bar(range(frequency.shape[0]), frequency)
plt.xticks(range(frequency.shape[0]), [str(i/10)+"-"+str((i+1)/10) for i in range(frequency.shape[0])])
plt.title("capacity reduction distribution")
plt.show()

frequency_capacity = pd.DataFrame({"capacity": bins[1:], "frequency": frequency})
frequency_capacity.to_csv("ScenarioAnalysis/df_frequencies/capacity_frequency.csv", index_label=False, index=False)

#
# regs_active = []
# regs_n_flights = []
# for i in range(0, 1440, 60):
#     reg_hour = reg_airp[((reg_airp.min_start <= i) & (i+60 <= reg_airp.min_end)) | ((i <= reg_airp.min_start) & (reg_airp.min_start <= i+60 ))
#                       | ((i <= reg_airp.min_end) & (reg_airp.min_end <= i + 60))]
#     regs_active.append(reg_hour.shape[0])
#     regs_n_flights.append(reg_hour.n_flights.mean())
#
# plt.bar(range(24),regs_active)
# plt.show()
#
# plt.bar(range(24),regs_n_flights)
# plt.show()
#
# reg_airp_freq = reg_airp.ReferenceLocationName.value_counts()
#
#
#
#





reg_lszh = reg[reg.ReferenceLocationName == "LSZH"]
regs_active = []
regs_n_flights = []
for i in range(0, 1440, 60):
    reg_hour = reg_lszh[((reg_lszh.min_start <= i) & (i+60 <= reg_lszh.min_end)) | ((i <= reg_lszh.min_start) & (reg_lszh.min_start <= i+60 ))
                      | ((i <= reg_lszh.min_end) & (reg_lszh.min_end <= i + 60))]
    regs_active.append(reg_hour.shape[0])
    regs_n_flights.append(reg_hour.n_flights.mean())

plt.bar(range(24),regs_active)
plt.show()

plt.bar(range(24),regs_n_flights)
plt.show()



# import CostPackage.arrival_costs as cp
#
# help(cp)
#
# data = cp.get_data_dict()
#
# data.keys()
# aircraft = data["aircraft"]
#
# air_clusters = aircraft.AssignedAircraftType.unique()
#
# for air in air_clusters:
#     delay = range(100)
#     cost_fun = cp.get_cost_model(aircraft_type=air, airline="rrr", destination="EBBF", n_passengers=11500)
#     plt.plot(delay, [cost_fun(d) for d in delay])
# plt.show()


cap = pd.read_csv("ScenarioAnalysis/RegulationCapacities/capacities.csv")
cap = cap[cap.ReferenceLocationName.isin(airports.Airport)]

mean_capacity = []
for i in range(0, 1440, 60):
    cap_hour = cap[((cap.min_start <= i) & (i+60 <= cap.min_end)) | ((i <= cap.min_start) & (cap.min_start <= i+60 ))
                      | ((i <= cap.min_end) & (cap.min_end <= i + 60))]
    mean_capacity.append(cap_hour.Capacity.mean())


max_capacity = []
for airport in cap.ReferenceLocationName.unique():
    max_capacity.append(max(cap[cap.ReferenceLocationName == airport].Capacity))

np.mean(max_capacity)


import pandas as pd
reg = pd.read_csv("ScenarioAnalysis/RegulationCapacities/regulations_.csv")
reg_25 = reg[reg.capacity_reduction >= 0.1]
airports = pd.read_csv("ScenarioAnalysis/airportMore25M.csv")
reg_25 = reg_25[reg_25.ReferenceLocationName.isin(airports.Airport)]

reg_25.to_csv("ScenarioAnalysis/RegulationCapacities/regulations_25_nozero.csv")

r = pd.read_csv("ScenarioAnalysis/RegulationCapacities/1907_regulations.csv")
r1 = pd.read_csv("ScenarioAnalysis/RegulationCapacities/1908_regulations.csv")

re = pd.concat([r,r1], ignore_index=True)