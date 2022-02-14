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


plt.scatter(reg[~reg.ReferenceLocationName.isin(airports)].capacity_reduction_mean,
            reg[~reg.ReferenceLocationName.isin(airports)].n_flights, s=15, c="blue", label="OTHER AIRPORTS")
plt.scatter(reg_25.capacity_reduction_mean, reg_25.n_flights, s=15, c="red", label="GROUP 1 AIRPORTS")
plt.ylabel("REGULATED FLIGHTS")
plt.xlabel("CAPACITY REDUCTION")
plt.legend()
plt.show()

plt.scatter(reg_25.capacity_reduction_mean, reg_25.n_flights, s=15, c="red")
plt.ylabel("REGULATED FLIGHTS")
plt.xlabel("CAPACITY REDUCTION")
plt.show()



# time 25

h_time = plt.hist(reg_25.min_start, bins=24, edgecolor='black', linewidth=1.2)
plt.grid(axis="y")
plt.xticks(h_time[1], range(25))
plt.ylabel("REGULATIONS")
plt.xlabel("TIME")
plt.show()



# capacity
h_cap = plt.hist(reg_25.capacity_reduction_mean, bins=[i*0.1 for i in range(11)], edgecolor='black', linewidth=1.2)
plt.grid(axis="y")
plt.xticks([i*0.1 for i in range(11)], [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.ylabel("REGULATIONS")
plt.xlabel("CAPACITY REDUCTION")
plt.show()

# flights
max(reg_25.n_flights)
plt.figure(figsize=(30, 15))
h_fl = plt.hist(reg_25.n_flights, bins=[10*i for i in range(41)], edgecolor='black', linewidth=1.2)
plt.xticks(h_fl[1], h_fl[1], rotation="vertical")
plt.ylabel("REGULATIONS")
plt.xlabel("REGULATED FLIGHTS NUMBER")
plt.show()



#airport

airports_frequency = reg.value_counts("ReferenceLocationName")
airports_frequency_25 = reg_25.value_counts("ReferenceLocationName")


plt.bar(range(airports_frequency_25.values.shape[0]), airports_frequency_25.values)
plt.xticks(range(airports_frequency_25.values.shape[0]), airports_frequency_25.index)
plt.grid(axis="y")
plt.tick_params(axis='x', rotation=45)
plt.ylabel("REGULATIONS")
plt.show()


# time cap fl

(0.75 + 1 - min(reg_25.capacity_reduction_mean))**12.5
(0.5 + 1 - min(reg_25.capacity_reduction_mean))**12.5
max(reg_25.capacity_reduction_mean)
max(((reg_25.capacity_reduction_mean + 1 - min(reg_25.capacity_reduction_mean)))**12.5)

plt.scatter(reg_25.min_start, reg_25.n_flights,
            s=((reg_25.capacity_reduction_mean + 1 - min(reg_25.capacity_reduction_mean)))**12.5)
c_1 = plt.scatter([], [], s=4000, marker='o', color='#1f77b4')
c_075 = plt.scatter([], [], s=1046, marker='o', color='#1f77b4')
c_05 = plt.scatter([], [], s=151, marker='o', color='#1f77b4')
c_025 = plt.scatter([], [], s=151, marker='o', color='#1f77b4')
plt.legend((c_1, c_075, c_05, c_025), ('1.0\n\n', '0.75\n\n', '0.5\n\n', '0.25'), title="  CAPACITY\nREDUCTION", scatterpoints=1,
           loc=(1.05, 0.6), ncol=1, fontsize=28)
plt.xticks(range(0, 1441, 60), range(25))
plt.ylabel("REGULATED FLIGHTS")
plt.xlabel("TIME")
plt.show()

# 'upper right'

plt.scatter(reg_25.min_start, reg_25.capacity_reduction_mean, s=reg_25.n_flights)
plt.xticks(range(0, 1441, 60), range(25))
plt.show()


cpm = reg_25.capacity_reduction_mean
np.exp(cpm*10)

min(reg_25.capacity_reduction_mean + 1 - min(reg_25.capacity_reduction_mean))




# airline distr
import pandas as pd
airline = pd.read_csv("ScenarioAnalysis/df_frequencies/airport_airline_frequency.csv")

print(airline)





