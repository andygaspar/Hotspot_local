import pandas as pd
import numpy as np

flights = pd.read_csv("ScenarioAnalysis/Flights/flights_complete.csv")
flights_regulated = pd.read_csv("ScenarioAnalysis/Flights/flights_regulated.csv")

airports = pd.read_csv("ScenarioAnalysis/airportMore25M.csv")
flights = flights[flights.Destination.isin(airports.Airport)]

low_costs = pd.read_csv("ScenarioAnalysis/df_frequencies/2017-LCC.csv")

low_cost_flights = flights[flights.Company.isin(low_costs.airline)]
high_cost_flights = flights[~flights.Company.isin(low_costs.airline)]

aircraft_low_cost = low_cost_flights.aircraft_cluster.value_counts()
aircraft_high_cost = high_cost_flights.aircraft_cluster.value_counts()

df_low_frequency = pd.DataFrame({"aircraft": aircraft_low_cost.index, "frequency": aircraft_low_cost.to_numpy()})
df_high_frequency = pd.DataFrame({"aircraft": aircraft_high_cost.index, "frequency": aircraft_high_cost.to_numpy()})

df_low_frequency.to_csv("ScenarioAnalysis/df_frequencies/aircraft_low.csv", index_label=False, index=False)
df_high_frequency.to_csv("ScenarioAnalysis/df_frequencies/aircraft_high.csv", index_label=False, index=False)




# df_air_frequency_filtered.to_csv("Aircraft/aircraft_frequency.csv", index_label=False, index=False)

df_airport_air_cluster_frequencies = pd.DataFrame(columns=["airport", "airline", "air_cluster", "frequency"])
for airport in airports.Airport:
    df_airport = flights[flights.Destination == airport]
    for airline in df_airport.Company.unique():
        df_airport_airline = df_airport[df_airport.Company == airline]
        air_cluster_frequency = df_airport_airline.aircraft_cluster.value_counts()
        n_entries = air_cluster_frequency.index.shape[0]
        df_airport_append = pd.DataFrame(
            {"airport": [airport for _ in range(n_entries)], "airline": [airline for _ in range(n_entries)],
             "air_cluster": air_cluster_frequency.index,
             "frequency": air_cluster_frequency.to_numpy()
             })
        df_airport_air_cluster_frequencies = pd.concat([df_airport_air_cluster_frequencies, df_airport_append],
                                                       ignore_index=True)


egll = df_airport_air_cluster_frequencies[(df_airport_air_cluster_frequencies.airport == "EGLL")]
dlh = egll[egll.airline == "DLH"]


df_airport_air_cluster_frequencies.to_csv("ScenarioAnalysis/df_frequencies/airport_airline_cluster_frequency.csv", index_label=False,
                               index=False)

