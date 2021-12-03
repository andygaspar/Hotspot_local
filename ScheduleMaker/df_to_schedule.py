from typing import Union, List, Callable
from CostPackage.arrival_costs import get_cost_model, get_data_dict
from ModelStructure.Costs.costFunctionDict import CostFuns
from ModelStructure.Slot.slot import Slot
from ModelStructure import modelStructure
from ModelStructure.Flight import flight as fl
from Istop.Preferences import preference
import numpy as np
import pandas as pd

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


class RealisticSchedule:

    def __init__(self):
        self.df_airline = pd.read_csv("ScenarioAnalysis/df_frequencies/airport_airline_frequency.csv")
        self.df_aircraft_high = pd.read_csv("ScenarioAnalysis/df_frequencies/aircraft_high.csv")
        self.df_aircraft_low = pd.read_csv("ScenarioAnalysis/df_frequencies/aircraft_low.csv")
        self.aircraft_seats = get_data_dict()["aircraft_seats"]
        self.df_capacity = pd.read_csv("ScenarioAnalysis/df_frequencies/airport_max_capacity.csv")
        self.pax = pd.read_csv("ScenarioAnalysis/Pax/pax.csv")

    def make_sl_fl_from_data(self, n_flights: int, capacity_reduction: float, load_factor=0.89, compute=True):

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

            passengers = int(self.aircraft_seats[self.aircraft_seats.Aircraft == fl_type]["SeatsLow"].iloc[0]
                             * load_factor)
            pax = self.pax[(self.pax.destination == airport) & (self.pax.airline == airline)]
            if pax.shape[0] > 0:
                pax = pax.sample(n=passengers, weights=pax.pax, replace=True)
                pax = pax[pax.leg2 > 0]
                if pax.shape[0] > 0:
                    missed_connected = pax.apply(lambda x: (x.delta_leg1, x.delay), axis=1).to_list()
                else:
                    missed_connected = None
            else:
                missed_connected = None
            cost_fun = get_cost_model(fl_type, airline, airport, passengers, missed_connected=missed_connected)
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

# make_sl_fl_from_data(5, 0.2)
