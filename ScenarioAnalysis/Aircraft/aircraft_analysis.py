import pandas as pd
import numpy as np

flights = pd.read_csv("ScenarioAnalysis/Flights/flights.csv")
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
