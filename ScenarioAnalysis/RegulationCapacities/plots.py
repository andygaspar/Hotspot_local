import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (25, 18)
plt.rcParams.update({'font.size': 10})


airports = pd.read_csv("ScenarioAnalysis/airportMore25M.csv").Airport
reg = pd.read_csv("ScenarioAnalysis/RegulationCapacities/regulations.csv")
reg_25 = reg[reg.ReferenceLocationName.isin(airports)]

airports_frequency_all = reg.value_counts("ReferenceLocationName")
airports_frequency_25_all = reg_25.value_counts("ReferenceLocationName")

reg = reg[(reg.capacity_reduction > 0)]
reg = reg[(reg.capacity_reduction_mean > 0)]
reg_25 = reg_25[(reg_25.capacity_reduction > 0)]
reg_25 = reg_25[(reg_25.capacity_reduction_mean > 0)]


plt.scatter(reg.capacity_reduction, reg.n_flights, s=15)
plt.show()

plt.scatter(reg.capacity_reduction_mean, reg.n_flights, s=15)
plt.show()

# time
h_time = plt.hist(reg.min_start, bins=24)
plt.xticks(range(0, 1441, 60), range(25))
plt.show()


# 25 *****************************




plt.scatter(reg_25.capacity_reduction_mean, reg_25.n_flights, s=15)
plt.show()

# capacity
h_cap = plt.hist(reg_25.capacity_reduction_mean, bins=[i*0.1 for i in range(11)])
plt.xticks([i*0.1 for i in range(11)], range(11))
plt.show()

# flights
max(reg_25.n_flights)
h_fl = plt.hist(reg_25.n_flights, bins=[10*i for i in range(41)])
plt.xticks(h_fl[1], h_fl[1])
plt.show()

airports_frequency = reg.value_counts("ReferenceLocationName")
airports_frequency_25 = reg_25.value_counts("ReferenceLocationName")


plt.scatter(reg_25.min_start, reg_25.n_flights, s=reg_25.capacity_reduction_mean*500)
plt.xticks(range(0, 1441, 60),range(25))
plt.show()

plt.scatter(reg_25.min_start, reg_25.capacity_reduction_mean, s=reg_25.n_flights)
plt.xticks(range(0, 1441, 60),range(25))
plt.show()






