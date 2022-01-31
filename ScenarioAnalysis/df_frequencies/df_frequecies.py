import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (25, 18)
plt.rcParams.update({'font.size': 10})


airports = pd.read_csv("ScenarioAnalysis/airportMore25M.csv").Airport
reg = pd.read_csv("ScenarioAnalysis/RegulationCapacities/regulations.csv")
reg_25 = reg[reg.ReferenceLocationName.isin(airports)]

reg = reg[(reg.capacity_reduction > 0)]
reg = reg[(reg.capacity_reduction_mean > 0)]
reg_25 = reg_25[(reg_25.capacity_reduction > 0)]
reg_25 = reg_25[(reg_25.capacity_reduction_mean > 0)]

# reg_25.to_csv("ScenarioAnalysis/RegulationCapacities/regulations_25_nozero.csv", index_label=False, index=False)

airports_frequency_all = reg.value_counts("ReferenceLocationName")
airports_frequency_25_all = reg_25.value_counts("ReferenceLocationName")


# number regulations per airport

df_25_reg_frequency = pd.DataFrame({"airport": airports_frequency_25_all.index, "n_regulations": airports_frequency_25_all.values})
# noinspection PyTypeChecker
# df_25_reg_frequency.to_csv("ScenarioAnalysis/df_frequencies/airports_regulations.csv", index_label=False, index=False)


# regulation time distribution

h_time = plt.hist(reg_25.min_start, bins=range(0, 1441,60))
df_time_reg_frequency = pd.DataFrame({"time": h_time[1][:-1], "n_regulations": h_time[0].astype(int)})
# noinspection PyTypeChecker
# df_time_reg_frequency.to_csv("ScenarioAnalysis/df_frequencies/time_regulation_freq.csv", index_label=False, index=False)




# capacity
h_cap = plt.hist(reg_25.capacity_reduction_mean, bins=[i*0.1 for i in range(11)])
df_cap_frequency = pd.DataFrame({"reduction": h_cap[1][:-1], "n_regualation": h_cap[0].astype(int)})
# noinspection PyTypeChecker
# df_cap_frequency.to_csv("ScenarioAnalysis/df_frequencies/capacity_frequency.csv", index_label=False, index=False)




# flights
max(reg_25.n_flights)
h_fl = plt.hist(reg_25.n_flights, bins=[10*i for i in range(41)])
df_n_flights = pd.DataFrame({"n_flights": h_fl[1][:-1], "frequency": h_fl[0].astype(int)})
# noinspection PyTypeChecker
# df_n_flights.to_csv("ScenarioAnalysis/df_frequencies/n_flights_frequency.csv", index_label=False, index=False)







