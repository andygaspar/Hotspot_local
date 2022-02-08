import pandas as pd

flights = pd.read_csv("ScenarioAnalysis/Flights/flights_complete.csv")


def get_flight_length(airline: str, airport: str, air_cluster: str):
    df_flights = flights[(flights.Destination == airport)
                         & (flights.Company == airline) & (flights.aircraft_cluster == air_cluster)]
    if df_flights.shape[0] == 0:
        print("flight not found")
        return 4000
    else:
        return df_flights.FlightLength.sample().iloc[0]
