from typing import List

from Istop.AirlineAndFlight.istopFlight import IstopFlight

from Istop.old.bb import BB, Offer
from Istop.old import bb

stop = bb.stop

class BB_new_2(BB):

    def __init__(self, offers, reductions, flights: List[IstopFlight], min_lp_len=80, max_lp_time=10, print_info=100):
        super().__init__(offers, reductions, flights, min_lp_len, max_lp_time, print_info)

        self.precomputed_len = 0
        self.max_precomputed = 0

    def set_match_for_flight(self, flights: List[IstopFlight]):
        for flight in flights:
            for offer in self.offers:
                match = offer.offer
                for couple in match:
                    if flight.slot == couple[0].slot or flight.slot == couple[1].slot:
                        flight.offers.append(offer)

    def run(self):
        self.step([], self.offers, 0)

        if len(self.solution) > 0:
            self.solution = [offer.offer for offer in self.solution]

    def step(self, solution: List[Offer], offers: list[Offer], reduction: float):

        # if self.nodes % self.info == 0:
        #     print(self.nodes, len(self.precomputed), self.stored, self.precomputed_len, self.max_precomputed)

        self.nodes += 1
        if len(offers) == 0:
            self.initSolution = True
            return 0

        l_reduction = reduction + offers[0].reduction
        l_solution = solution + [offers[0]]

        if l_reduction > self.best_reduction:
            self.solution = l_solution
            self.best_reduction = l_reduction

        l_incompatible = [offer for flight in offers[0].flights for offer in flight.offers]
        l_offers = [offer for offer in offers[1:] if offer not in l_incompatible]
        l_offers_key = ".".join([str(offer.num) for offer in l_offers])

        pruned = False
        if self.initSolution:
            if l_offers_key in self.precomputed.keys():
                if self.precomputed[l_offers_key] + l_reduction < self.best_reduction:
                    # self.stored += 1
                    # self.precomputed_len = (self.precomputed_len * (self.stored - 1) + len(l_offers))/self.stored
                    # if self.max_precomputed < len(l_offers):
                    #     self.max_precomputed = len(l_offers)
                    best_left = self.precomputed[l_offers_key]
                    pruned = True

            else:
                l_offers_reduction = sum([offer.reduction for offer in l_offers])
                bound = l_reduction + l_offers_reduction
                if bound < self.best_reduction:
                    pruned = True
                    best_left = l_offers_reduction

        if not pruned:

            best_left = self.step(l_solution, l_offers, l_reduction)

        r_offers = offers[1:]
        r_offers_key = ".".join([str(offer.num) for offer in r_offers])

        pruned = False
        if r_offers_key in self.precomputed.keys():
            if self.precomputed[r_offers_key] + reduction < self.best_reduction:
                # self.stored += 1
                # self.precomputed_len = (self.precomputed_len * (self.stored - 1) + len(r_offers))/self.stored
                # if self.max_precomputed < len(r_offers):
                #     self.max_precomputed = len(r_offers)
                best_right = self.precomputed[r_offers_key]
                pruned = True
        else:
            r_offers_reduction = sum([offer.reduction for offer in r_offers])
            bound = reduction + r_offers_reduction
            if bound < self.best_reduction:
                pruned = True
                best_right = r_offers_reduction

        if not pruned:
            best_right = self.step(solution, r_offers, reduction)

        best = max(best_left + offers[0].reduction, best_right)
        self.precomputed[str(offers[0].num) + "." + r_offers_key] = best

        return best


