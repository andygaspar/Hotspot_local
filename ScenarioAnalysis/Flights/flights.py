import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

flights = pd.read_csv("ScenarioAnalysis/Flights/flights.csv")
airports = pd.read_csv("ScenarioAnalysis/airportMore25M.csv")
flights = flights[flights.Destination.isin(airports.Airport)]

airlines = flights.Company.unique()

airline_frequency = flights.Company.value_counts()
low_costs = pd.read_csv("ScenarioAnalysis/Flights/2017-LCC.csv")

df_frequency = pd.DataFrame({"airline": airline_frequency.index, "frequency": airline_frequency.to_numpy(),
                             "low_cost": [True if airline in list(low_costs.airline) else False
                                          for airline in airline_frequency.index]})

df_frequency.to_csv("ScenarioAnalysis/df_frequencies/airline_frequency.csv", index_label=False, index=False)

df_airports_frequencies = pd.DataFrame(columns=["airport", "airline", "frequency", "low_cost"])
for airport in airports.Airport:
    df_airport = flights[flights.Destination == airport]
    airp_freq = df_airport.Company.value_counts()
    df_airport_append = pd.DataFrame(
        {"airport": [airport for _ in range(airp_freq.index.shape[0])], "airline": airp_freq.index,
         "frequency": airp_freq.to_numpy(),
         "low_cost": [True if airline in list(low_costs.airline) else False
                      for airline in airp_freq.index]})
    df_airports_frequencies = pd.concat([df_airports_frequencies, df_airport_append], ignore_index=True)

df_airports_frequencies.to_csv("ScenarioAnalysis/df_frequencies/airport_airline_frequency.csv", index_label=False,
                               index=False)
