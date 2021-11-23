import pandas as pd
import numpy as np
import csv


def time_to_int(t):
    hh = int(t[11:13])
    mm = int(t[14:16])
    return mm + hh * 60


def set_dfs():
    cap1 = pd.read_csv("ScenarioAnalysis/RegulationCapacities/1907_capacities.csv")
    cap1["month"] = [7 for _ in range(cap1.shape[0])]

    cap2 = pd.read_csv("ScenarioAnalysis/RegulationCapacities/1908_capacities.csv")
    cap1["month"] = [8 for _ in range(cap1.shape[0])]

    cap = pd.concat([cap1, cap2], ignore_index=True)
    cap = cap.replace(to_replace="EBBR/MB", value="EBBR")
    cap = cap[cap.ReferenceLocationRole == "A"]

    days = cap.StartTimestamp.apply(lambda d: d[:10]).unique()
    days_dict = dict(zip(days, range(days.shape[0])))
    cap["day_start"] = cap.StartTimestamp.apply(lambda d: days_dict[d[:10]])
    cap["day_end"] = cap.EndTimestamp.apply(lambda d: days_dict[d[:10]])
    cap["min_start"] = cap.StartTimestamp.apply(time_to_int)
    cap["min_end"] = cap.EndTimestamp.apply(time_to_int)
    cap["start"] = cap.day_start * 60 * 24 + cap.min_start
    cap["end"] = cap.day_end * 60 * 24 + cap.min_end

    reg1 = pd.read_csv("ScenarioAnalysis/RegulationCapacities/1907_regulations.csv")
    reg1["month"] = [7 for _ in range(reg1.shape[0])]

    reg2 = pd.read_csv("ScenarioAnalysis/RegulationCapacities/1908_regulations.csv")
    reg2["month"] = [8 for _ in range(reg2.shape[0])]

    reg = pd.concat([reg1, reg2], ignore_index=True)
    reg = reg[reg.ReferenceLocationRole == "A"]
    reg = reg.replace(to_replace="EBBR/MB", value="EBBR")

    days = reg.StartTimestamp.apply(lambda d: d[:10]).unique()
    days_dict = dict(zip(days, range(days.shape[0])))

    reg["day_start"] = reg.StartTimestamp.apply(lambda d: days_dict[d[:10]])
    reg["day_end"] = reg.EndTimestamp.apply(lambda d: days_dict[d[:10]])
    reg["min_start"] = reg.StartTimestamp.apply(time_to_int)
    reg["min_end"] = reg.EndTimestamp.apply(time_to_int)
    reg["start"] = reg.day_start * 60 * 24 + reg.min_start
    reg["end"] = reg.day_end * 60 * 24 + reg.min_end

    reg["duration"] = reg.end - reg.start
    reg = reg[reg.duration > 0]

    initial_capacities = []

    for r in reg.RegulationID:
        try:
            regulation = reg[reg.RegulationID == r]
            location = regulation.ReferenceLocationName.iloc[0]
            start = regulation.start.iloc[0]
            initial_capacity = max(
                cap[(cap.ReferenceLocationName == location) & (cap.start <= start) & (start <= cap.end)].Capacity)
            initial_capacities.append(initial_capacity)

        except:
            initial_capacities.append(0)

    reg["initial_capacity"] = initial_capacities
    reg = reg[(reg.initial_capacity > 0) & (reg.Capacity > 0)]
    reg["capacity_reduction"] = (reg.initial_capacity - reg.Capacity) / reg.initial_capacity

    flights = pd.read_csv("ScenarioAnalysis/Flights/flights_regulated.csv")
    flights_delayed = flights[flights.ATFMDelay > 0]
    n_flights = []
    for r in reg.RegulationID:
        n_flights.append(flights_delayed[flights_delayed.MPR == r].shape[0])

    reg["n_flights"] = n_flights

    reg.to_csv("ScenarioAnalysis/RegulationCapacities/regulations.csv", index_label=False, index=False)
    cap.to_csv("ScenarioAnalysis/RegulationCapacities/capacities.csv", index_label=False, index=False)


def get_airport_set_dicts():
    airports = pd.read_csv("ScenarioAnalysis/airportMore25M.csv")
    airport_set1 = {}
    with open("ScenarioAnalysis/RegulationCapacities/1907_set_of_airports") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            airport_set1[row[0]] = row[1:]

    airport_set2 = {}
    with open("ScenarioAnalysis/RegulationCapacities/1908_set_of_airports") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            airport_set2[row[0]] = row[1:]

    airport_set1_filtered = {}
    for key in airport_set1.keys():
        if not set(airport_set1[key]).isdisjoint(set(airports.Airport)):
            airport_set1_filtered[key] = airport_set1[key]

    airport_set2_filtered = {}
    for key in airport_set2.keys():
        if not set(airport_set2[key]).isdisjoint(set(airports.Airport)):
            airport_set2_filtered[key] = airport_set2[key]

    return airport_set1_filtered, airport_set2_filtered

set_dfs()
