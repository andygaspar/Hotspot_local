from typing import Union, List, Callable
from CostPackage.arrival_costs import get_cost_model, get_data_dict
from ModelStructure.Costs.costFunctionDict import CostFuns
from ModelStructure.Slot.slot import Slot
from ModelStructure import modelStructure
from ModelStructure.Flight import flight as fl
from Istop.Preferences import preference
import numpy as np
import pandas as pd

from ScenarioAnalysis.Curfew.curfew import get_curfew_threshold

df_west_m = pd.read_csv("ScenarioAnalysis/Flights/flight_schedules_westminster.csv")


fl_id_to_type = dict(zip(df_west_m.nid, df_west_m.aircraft_type))
cost_funs = CostFuns()


def make_flight(line, slot_times, df_costs):
    slot_index = line["slot"]
    flight_name = line["flight"]
    airline_name = line["airline"]
    eta = line["eta"]
    slot_time = line['fpfs']
    fl_type = None

    # slot = Slot(slot_index, slot_time)
    if df_costs is None:

        delay_cost_vect, fl_id = cost_funs.get_random_cost_vect(slot_times, eta)
        fl_type = fl_id_to_type[fl_id]
    else:
        delay_cost_vect = df_costs[flight_name]

    print(fl_id, fl_type)

    return modelStructure.make_slot_and_flight(slot_time=slot_time, slot_index=slot_index, eta=eta,
                                               flight_name=flight_name, airline_name=airline_name,
                                               delay_cost_vect=delay_cost_vect, fl_type=fl_type)


def make_flight_list(df: pd.DataFrame, df_costs: pd.DataFrame = None):
    slot_list = []
    flight_list = []
    slot_times = df.time.to_numpy()
    for i in range(df.shape[0]):
        line = df.iloc[i]
        slot, flight = make_flight(line, slot_times, df_costs)
        slot_list.append(slot)
        flight_list.append(flight)

    return slot_list, flight_list


class Regulation:
    def __init__(self, airport, n_flights, c_reduction, start_time):
        self.airport = airport
        self.nFlights = n_flights
        self.cReduction = np.around(c_reduction, decimals=1)
        self.startTime = start_time

class RealisticSchedule:

    def __init__(self):
        self.df_airline = pd.read_csv("ScenarioAnalysis/df_frequencies/airport_airline_frequency.csv")
        self.df_aircraft_high = pd.read_csv("ScenarioAnalysis/df_frequencies/aircraft_high.csv")
        self.df_aircraft_low = pd.read_csv("ScenarioAnalysis/df_frequencies/aircraft_low.csv")
        self.aircraft_seats = get_data_dict()["aircraft_seats"]
        self.df_capacity = pd.read_csv("ScenarioAnalysis/df_frequencies/airport_max_capacity.csv")
        self.pax = pd.read_csv("ScenarioAnalysis/Pax/pax.csv")
        self.df_turnaround = pd.read_csv('ScenarioAnalysis/Aircraft/turnaround.csv')

        self.turnaround_dict = dict(zip(self.df_turnaround.AirCluster, self.df_turnaround.MinTurnaround))

        self.regulations = pd.read_csv("ScenarioAnalysis/RegulationCapacities/regulations_25_nozero.csv")

    def make_sl_fl_from_data(self, n_flights: int, capacity_reduction: float, load_factor=0.89,
                             regulation_time: int = 0, compute=True):

        airport = np.random.choice(self.df_airline.airport.to_list())
        df_airline = self.df_airline[self.df_airline.airport == airport]
        capacity = self.df_capacity[self.df_capacity.airport == airport].capacity.iloc[0]

        interval = 60 / capacity
        new_interval = 60 / (capacity * (1 - capacity_reduction))
        times = np.linspace(0, n_flights * interval, n_flights)
        new_times = np.linspace(0, n_flights * new_interval, n_flights)

        airline_low = df_airline[df_airline.low_cost].airline.to_list()

        flight_list = []
        slot_list = []

        for i in range(n_flights):
            airline = df_airline.airline.sample(weights=df_airline.frequency).iloc[0]
            # print(airline)
            if airline in airline_low:
                fl_type = self.df_aircraft_low.aircraft.sample(weights=self.df_aircraft_low.frequency).iloc[0]
            else:
                fl_type = self.df_aircraft_high.aircraft.sample(weights=self.df_aircraft_high.frequency).iloc[0]

            passengers = self.get_passengers(airport=airport, airline=airline,
                                             air_cluster=fl_type, load_factor=load_factor)
            pax_connections = self.pax[(self.pax.destination == airport) & (self.pax.airline == airline)]
            if pax_connections.shape[0] > 0:
                pax_connections = pax_connections.sample(n=passengers, weights=pax_connections.pax, replace=True)
                pax_connections = pax_connections[pax_connections.leg2 > 0]
                if pax_connections.shape[0] > 0:
                    missed_connected = pax_connections.apply(lambda x: (x.delta_leg1, x.delay), axis=1).to_list()
                else:
                    missed_connected = None
            else:
                missed_connected = None

            eta = regulation_time + times[i]
            min_turnaround = self.turnaround_dict[fl_type]
            curfew_th, rotation_destination = get_curfew_threshold(airport, airline, fl_type, eta, min_turnaround)
            react_curfew = (curfew_th, self.get_passengers(rotation_destination, airline, fl_type, load_factor)) \
                if curfew_th is not None else None

            cost_fun = get_cost_model(fl_type, airline, airport, passengers, missed_connected=missed_connected,
                                      react_curfew = react_curfew)
            delay_cost_vect = np.array([cost_fun(new_times[j]) for j in range(n_flights)])
            if compute:
                slot, flight = modelStructure.make_slot_and_flight(slot_time=new_times[i], slot_index=i, eta=times[i],
                                                                   flight_name=airline + str(i), airline_name=airline,
                                                                   delay_cost_vect=delay_cost_vect, fl_type=fl_type)
                slot_list.append(slot)
                flight.costFun = cost_fun
                flight_list.append(flight)

        return slot_list, flight_list, airport
        # slot: Slot, flight_name: str, airline_name: str,
        # eta: float, delay_cost_vect = np.array,

    def get_passengers(self, airport, airline, air_cluster, load_factor):
        pax = self.pax[(self.pax.destination == airport)
                       & (self.pax.airline == airline)
                       & (self.pax.air_cluster == air_cluster)]
        if pax.shape[0] > 0:
            flight_sample = pax.leg1.sample().iloc[0]
            passengers = pax[pax.leg1 == flight_sample].pax.sum()
        else:
            passengers = int(self.aircraft_seats[self.aircraft_seats.Aircraft == air_cluster]["SeatsLow"].iloc[0]
                             * load_factor)
        return passengers

    def get_regulation(self, capacity_min=0., n_flights_min=0, n_flights_max=1000, start=0, end=1441):
        regulations = self.regulations[(self.regulations.capacity_reduction_mean >= capacity_min) &
                                       (self.regulations.n_flights >= n_flights_min) &
                                       (self.regulations.n_flights <= n_flights_max) &
                                       (self.regulations.min_start >= start) &
                                       (self.regulations.min_end <= end)]
        regulation = regulations.sample().iloc[0]
        regulation = Regulation(airport=regulation.ReferenceLocationName, n_flights=regulation.n_flights,
                                c_reduction=regulation.capacity_reduction_mean,
                                start_time=regulation.min_start)
        return regulation