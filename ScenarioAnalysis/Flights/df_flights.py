import pandas as pd
import numpy as np

aircraft_cluster = pd.read_csv("ScenarioAnalysis/Aircraft/aircraftClustering.csv")
cluster_dict = dict(zip(aircraft_cluster.AircraftType, aircraft_cluster.AssignedAircraftType))

fl1 = pd.read_csv("ScenarioAnalysis/Flights/1907_flights.csv")
fl2 = pd.read_csv("ScenarioAnalysis/Flights/1908_flights.csv")
flights = pd.concat([fl1, fl2], ignore_index=True)
flights = flights[flights.AircraftType.isin(aircraft_cluster.AssignedAircraftType)]
flights["aircraft_cluster"] = flights.AircraftType.apply(lambda a: cluster_dict[a])

flights.to_csv("ScenarioAnalysis/Flights/flights.csv", index_label=False, index=False)


flights_regulated = flights[flights.MPR != "X"]

low_costs = pd.read_csv("ScenarioAnalysis/df_frequencies/2017-LCC.csv")

flights_regulated = flights_regulated[flights_regulated.AircraftType.isin(aircraft_cluster.AircraftType)]
airports = pd.read_csv("ScenarioAnalysis/airportMore25M.csv")
flights_regulated = flights_regulated[flights_regulated.Destination.isin(airports.Airport)]
flights_regulated["low_cost"] = flights_regulated.Company.apply(lambda airline: airline in list(low_costs.airline))
flights_regulated.to_csv("ScenarioAnalysis/Flights/flights_regulated.csv", index_label=False, index=False)