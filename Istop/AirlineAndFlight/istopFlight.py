import numpy as np

from ModelStructure.Flight import flight as fl
from Istop.Preferences import preference


class IstopFlight(fl.Flight):

    def __init__(self, flight: fl.Flight):

        super().__init__(*flight.get_attributes())

        self.priority = None

        self.fitCostVect = self.costVect

        self.flight_id = None

        self.standardisedVector = None

    def set_automatic_preference_vect(self, max_delay):
        self.slope, self.margin1, self.jump2, self.margin2, self.jump2 = \
            preference.make_preference_fun(max_delay, self.delayCostVect)

    def not_paramtrised(self):
        return self.slope == self.margin1 == self.jump2 == self.margin2 == self.jump2 is None

    def set_fit_vect(self):

        if self.slope == self.margin1 == self.jump1 == self.margin2 == self.jump2 is None:
            self.fitCostVect = self.costVect

        elif self.slope is not None and self.margin1 == self.jump1 == self.margin2 == self.jump2 is None:
            self.fitCostVect = preference.approx_linear(self.delayVect, self.slope)

        elif self.slope == self.margin1 == self.jump1 is not None and self.margin2 == self.jump2 is None:
            self.fitCostVect = preference.approx_slope_one_margin(self.delayVect, self.slope, self.margin1, self.jump1)

        elif self.slope == self.margin1 == self.jump1 == self.margin2 == self.jump2 is not None:
            self.fitCostVect = preference.approx_slope_two_margins(
                self.delayVect, self.slope, self.margin1, self.jump1, self.margin2, self.jump2)

        elif self.slope == self.margin2 == self.jump2 is None and self.margin1 == self.jump1  is not None:
            self.fitCostVect = preference.approx_one_margins(self.delayVect, self.margin1, self.jump1)

        elif self.slope is None and self.margin1 == self.jump1 == self.margin2 == self.jump2 is not None:
            self.fitCostVect = preference.approx_two_margins(
                self.delayVect, self.margin1, self.jump1, self.margin2, self.jump2)

