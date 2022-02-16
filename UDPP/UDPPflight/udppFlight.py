from ModelStructure.Flight import flight as fl


class UDPPflight(fl.Flight):

    def __init__(self, flight: fl.Flight):
        super().__init__(slot=flight.slot, flight_name=flight.name, airline_name=flight.airlineName, eta=flight.eta,
                         delay_cost_vect=flight.delayCostVect, udpp_priority=flight.udppPriority,
                         udpp_priority_number=flight.udppPriorityNumber, tna=flight.tna, slope=flight.slope,
                         margin_1=flight.margin1, jump_1=flight.jump1, margin_2=flight.margin2, jump_2=flight.jump1,
                         fl_type=flight.type, missed_connecting=flight.missed_connecting, curfew=flight.curfew)

        # UDPP attributes ***************

        self.UDPPLocalSlot = None

        self.UDPPlocalSolution = None

        self.test_slots = []

        self.localTime = None

    def set_prioritisation(self, udpp_priority: str, udpp_priority_number: int = None, tna: float = None):
        self.udppPriority = udpp_priority
        self.udppPriorityNumber = udpp_priority_number
        self.tna = tna
