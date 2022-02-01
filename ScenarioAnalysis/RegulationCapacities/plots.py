import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

plt.rcParams["figure.figsize"] = (25, 18)
plt.rcParams.update({'font.size': 25})
plt.rcParams["figure.autolayout"] = True

flights = pd.read_csv("ScenarioAnalysis/Flights/flights_complete.csv")
flights.shape[0]
des = flights.Destination.unique()
ori = flights.Origin.unique()
airp = np.concatenate([des, ori])
np.unique(airp).shape

pax = pd.read_csv("ScenarioAnalysis/Pax/pax.csv")
fl_uow = pd.read_csv("ScenarioAnalysis/Flights/flight_schedules_westminster.csv")

pax[pax.leg2>-1].pax.sum()

des = pax.destination.unique()
ori = pax.origin.unique()
airp = np.concatenate([des, ori])
np.unique(airp).shape

airports = pd.read_csv("ScenarioAnalysis/airportMore25M.csv").Airport
reg = pd.read_csv("ScenarioAnalysis/RegulationCapacities/regulations.csv")
reg_25 = reg[reg.ReferenceLocationName.isin(airports)]

airports_frequency_all = reg.value_counts("ReferenceLocationName")
airports_frequency_25_all = reg_25.value_counts("ReferenceLocationName")

reg = reg[(reg.capacity_reduction > 0)]
reg = reg[(reg.capacity_reduction_mean > 0)]
reg_25 = reg_25[(reg_25.capacity_reduction > 0)]
reg_25 = reg_25[(reg_25.capacity_reduction_mean > 0)]
reg_25 = reg_25[reg_25.n_flights > 0]

reg.n_flights.sum()
reg_25.n_flights.sum()


plt.scatter(reg[~reg.ReferenceLocationName.isin(airports)].capacity_reduction_mean,reg[~reg.ReferenceLocationName.isin(airports)].n_flights, s=15, c="blue")
plt.scatter(reg_25.capacity_reduction_mean, reg_25.n_flights, s=15, c="red")
plt.show()

plt.scatter(reg_25.capacity_reduction_mean, reg_25.n_flights, s=15, c="red")
plt.show()



# time 25

h_time = plt.hist(reg_25.min_start, bins=24, edgecolor='black', linewidth=1.2)
plt.grid(axis="y")
plt.xticks(h_time[1], range(25))
plt.show()



# capacity
h_cap = plt.hist(reg_25.capacity_reduction_mean, bins=[i*0.1 for i in range(11)], edgecolor='black', linewidth=1.2)
plt.grid(axis="y")
plt.xticks([i*0.1 for i in range(11)], [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.show()

# flights
max(reg_25.n_flights)
plt.figure(figsize=(30, 15))
h_fl = plt.hist(reg_25.n_flights, bins=[10*i for i in range(41)], edgecolor='black', linewidth=1.2)
plt.xticks(h_fl[1], h_fl[1], rotation="vertical")
plt.show()



#airport

airports_frequency = reg.value_counts("ReferenceLocationName")
airports_frequency_25 = reg_25.value_counts("ReferenceLocationName")


plt.bar(range(airports_frequency_25.values.shape[0]), airports_frequency_25.values)
plt.xticks(range(airports_frequency_25.values.shape[0]), airports_frequency_25.index)
plt.grid(axis="y")
plt.tick_params(axis='x', rotation=45)
plt.show()


# time cap fl

plt.scatter(reg_25.min_start, reg_25.n_flights,
            s=((reg_25.capacity_reduction_mean + 1 - min(reg_25.capacity_reduction_mean)))**12.5)
plt.xticks(range(0, 1441, 60),range(25))
plt.show()

size = np.array([i*0.1 for i in range(1, 10)])
plt.scatter([0 for i in range(9)], size, s=(size+1)**12.5)
plt.show()


plt.scatter(reg_25.min_start, reg_25.capacity_reduction_mean, s=reg_25.n_flights)
plt.xticks(range(0, 1441, 60),range(25))
plt.show()


cpm = reg_25.capacity_reduction_mean
np.exp(cpm*10)

min(reg_25.capacity_reduction_mean + 1 - min(reg_25.capacity_reduction_mean))





