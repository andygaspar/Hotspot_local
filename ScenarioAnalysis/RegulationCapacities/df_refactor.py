import pandas as pd
import numpy as np
import csv


def time_to_int(t):
    hh = int(t[11:13])
    mm = int(t[14:16])
    return mm + hh * 60


def set_dfs():
    df_flights = pd.read_csv("ScenarioAnalysis/Flights/flights_complete.csv")
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
    cap.loc[cap.ReferenceLocationName == "EGLL", "Capacity"] = 46
    cap_corner = pd.read_csv("ScenarioAnalysis/RegulationCapacities/capacity_from_corner.csv")
    # cap = cap[~cap["ReferenceLocationName"].isin(cap_corner.ReferenceLocationName)]
    # cap = pd.concat([cap_corner, cap[["ReferenceLocationName", "Capacity"]]], ignore_index=True)

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
    actual_capacity = []
    actual_capacity_mean = []
    fl_delayed = []
    i = 0

    for r in reg.RegulationID:
        print(i)
        i += 1
        regulation = reg[reg.RegulationID == r]
        location = regulation.ReferenceLocationName.iloc[0]

        start = regulation.start.iloc[0]
        location_capacity = cap[(cap.ReferenceLocationName == location)]

        initial_capacity = max(location_capacity.Capacity) if location_capacity.shape[0] > 0 else 0

        initial_capacities.append(initial_capacity)
        fl_reg = df_flights[df_flights.MPR == r].sort_values(by="arr_min")
        n_flights = fl_reg.shape[0]
        if n_flights > 0:
            start, end = min(fl_reg.arr_min), max(fl_reg.arr_min)
            actual_cap = int(np.round(60 / ((end - start)/n_flights))) if end - start > 0 else initial_capacity

            mean_capacity = [int(np.round(60 / ((fl_reg.arr_min.iloc[i] - start)/(i + 1)))) if
                             (fl_reg.arr_min.iloc[i] - start) > 0
                             else initial_capacity for i in range(n_flights)]

            act_mean_capacity = np.mean(mean_capacity)


        else:
            actual_cap = initial_capacity
            act_mean_capacity = initial_capacity

        actual_capacity.append(actual_cap)
        actual_capacity_mean.append(act_mean_capacity)

        fl_delayed.append(df_flights[(df_flights.MPR == r) & (df_flights.ATFMDelay > 0)].shape[0])

    reg["initial_capacity"] = initial_capacities
    reg["actual_capacity"] = actual_capacity
    reg["actual_capacity_mean"] = actual_capacity_mean
    reg["n_flights"] = fl_delayed
    reg = reg[(reg.initial_capacity > 0) & (reg.Capacity > 0)]
    reg["capacity_reduction"] = (reg.initial_capacity - reg.actual_capacity) / reg.initial_capacity
    reg["capacity_reduction_mean"] = (reg.initial_capacity - reg.actual_capacity_mean) / reg.initial_capacity

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

