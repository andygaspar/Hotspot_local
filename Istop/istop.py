from typing import List

from Istop.AirlineAndFlight.istopFlight import IstopFlight
from Istop.Solvers.bb_bool import BBool
from Istop.Solvers.bb_p import TreeExplorer
from Istop.Solvers.gurobySolverOffer import GurobiSolverOffer
from Istop.old.bb_new_2 import BB_new_2
# from Istop.Solvers.mip_solver import MipSolver
# from Istop.Solvers.xpress_solver import XpressSolver
from Istop.old.bb_new_3 import BB_new_3
from ModelStructure import modelStructure as mS

import sys
from itertools import combinations
from Istop.AirlineAndFlight.istopAirline import IstopAirline
from ModelStructure.Flight.flight import Flight
from ModelStructure.Slot.slot import Slot
from OfferChecker.offerChecker import OfferChecker
import Istop.Solvers.bb_c as c_bool

import numpy as np
import pandas as pd

import time

# from Istop.Solvers.bb_p import TreeExplorer
import Istop.Solvers.bb_bool_parallel as bp

class Istop(mS.ModelStructure):

    @staticmethod
    def index(array, elem):
        for i in range(len(array)):
            if np.array_equiv(array[i], elem):
                return i

    def get_match_for_flight(self, flight):
        j = 0
        indexes = []
        for match in self.matches:
            for couple in match:
                if flight.slot == couple[0].slot or flight.slot == couple[1].slot:
                    indexes.append(j)
            j += 1
        return indexes

    def __init__(self, slot_list: List[Slot], flights: List[Flight], triples=False):
        self.offers = None
        self.triples = triples

        istop_flights = [IstopFlight(flight) for flight in flights]

        super().__init__(slot_list, istop_flights, air_ctor=IstopAirline)

        self.airlines: List[IstopAirline]

        max_delay = self.slots[-1].time - self.slots[0].time
        for flight in self.flights:
            flight.fitCostVect = flight.costVect

            if flight.not_paramtrised():
                flight.set_automatic_preference_vect(max_delay)

        for airline in self.airlines:
            airline.set_and_standardise_fit_vect()

        self.airlines_pairs = np.array(list(combinations(self.airlines, 2)))
        self.airlines_triples = np.array(list(combinations(self.airlines, 3)))

        self.epsilon = sys.float_info.min
        self.offerChecker = OfferChecker(self.scheduleMatrix)
        self.reductions = None

        self.matches = []
        self.couples = []
        self.flights_in_matches = []

        self.offers_selected = []

        self.problem = None

    def check_and_set_matches(self):
        start = time.time()
        self.matches = self.offerChecker.all_couples_check(self.airlines_pairs)
        if self.triples:
            self.matches += self.offerChecker.all_triples_check(self.airlines_triples)

        for match in self.matches:
            for couple in match:
                if not self.is_in(couple, self.couples):
                    self.couples.append(couple)
                    if not self.f_in_matched(couple[0]):
                        self.flights_in_matches.append(couple[0])
                    if not self.f_in_matched(couple[1]):
                        self.flights_in_matches.append(couple[1])

        self.reductions = self.offerChecker.get_reductions(self.matches)
        print("preprocess concluded in sec:", time.time() - start, "   Number of possible offers: ", len(self.matches))
        return len(self.matches) > 0

    def run(self, max_time=120, timing=False, verbose=False, branching=False):
        feasible = self.check_and_set_matches()
#match before [[array([IBK25, IBK34], dtype=object), array([SAS26, SAS30], dtype=object)], [array([RYR3, RYR17], dtype=object), array([SAS5, SAS22], dtype=object)], [array([SAS19, SAS30], dtype=object), array([VKG32, VKG18], dtype=object)]]
        # print("match before", self.matches)
        # t = time.time()
        # bbol = BBool(offers=self.matches, reductions=self.reductions, flights=self.flights, min_lp_len=5,
        #               print_info=1000000)
        # t = time.time() - t
        # print("prep time", t)
        # bool_cpp = c_bool.Run(bbol.compatibilityMatrix, bbol.reductions, bbol.offers)
        # bool_cpp.test()
        g_offer = GurobiSolverOffer(self, offers=self.matches, reductions=self.reductions)
        g_offer.run(timing=True)
        print("reduction gurobi ", g_offer.m.getObjective().getValue())



        # offers = bbol.offers
        # reductions = bbol.reductions
        # num_offers = len(offers)
        #
        # init_state = bp.State(np.full(num_offers, False), 0, reductions, bbol.compatibilityMatrix, False)
        # init_node = bp.Node(np.full(num_offers, False), np.full(num_offers, True), 0)
        #
        # t = time.time()
        # bb = TreeExplorer(init_state, init_node, bp.update_state, bp.process_node)
        # final_state = bb.explore_tree()
        # t = time.time()-t
        # print("time parallel", t, final_state.reduction)
        # print(final_state.solution)
        # for i, off in enumerate(bbol.offers):
        #     if final_state.solution[i]:
        #         print(off)



        # t = time.time()
        # bbol.run()
        # t = time.time() - t
        # print("\ntime alg", t, "nodes", bbol.nodes)
        # print("bb sol len ", len(bbol.solution))
        # print("bb reduction ", bbol.best_reduction)
        # for o in bbol.solution:
        #     print(o)
        #
        # print("\n")

        # bool_cpp = c_bool.Run(bbol.compatibilityMatrix, bbol.reductions, bbol.offers)
        #
        # bool_cpp.test()


        # print("dict")
        # t = time.time()
        # bb = BB_new_2(offers=self.matches, reductions=self.reductions, flights=self.flights, min_lp_len=5,
        #               print_info=5000)
        # t = time.time() - t
        #
        # print("prep time", t)
        #
        # t = time.time()
        #
        # bb.run()
        # t = time.time() - t
        # print("time alg", t, "nodes", bb.nodes, "stored", len(bb.precomputed), "found", bb.stored)
        # print("bb sol len ", len(bb.solution))
        # print("bb reduction ", bb.best_reduction)
        # for o in bb.solution:
        #     print(o)
        #
        # print("\n")













        # print("bb sol len ", len(bb.solution))
        # print("bb reduction ", bb.best_reduction)
        # for o in bb.solution:
        #     print(o)


        # if feasible:
        #     t = time.time()
        #     self.problem = GurobiSolver(self)
        #     solution_vect, offers_vect = self.problem.run(timing=timing, verbose=verbose, branching=branching)
        #     print("time Gurobi", time.time()-t)
        #
        #     self.assign_flights(solution_vect)
        #
        #     print("gurobi solution")
        #     offers = 0
        #     for i in range(len(self.matches)):
        #         if offers_vect[i] > 0.9:
        #             self.offers_selected.append(self.matches[i])
        #             # print(self.matches[i])
        #             offers += 1
        #     print("Number of offers selected: ", offers)
        #
        #
        #
        # else:
        #     for flight in self.flights:
        #         flight.newSlot = flight.slot
        #
        # solution.make_solution(self)
        # self.offer_solution_maker()
        # print("reduction", self.initialTotalCosts - self.compute_costs(self.flights, "final"))
        # 1438
        # print("start")
        # t = time.time()
        # bb = BBVisual(offers=self.matches, reductions=self.reductions, flights=self.flights, min_lp_len=1,
        #               print_info=5000, print_tree=5000)
        # bb.run()
        # print("time alg", time.time() - t, "nodes", bb.nodes, "stored", bb.stored)
        # print('lp time', bb.lp_time)

        # print("bb sol len ", len(bb.solution))
        # print("bb reduction ", bb.best_reduction)
        # print("solution \n")



        # print('lp time', bb.lp_time)

        # print("bb sol len ", len(bb.solution))
        # print("bb reduction ", bb.best_reduction)
        # print("solution \n")
        # for offer in bb.solution:
        #     print(offer)




    def other_airlines_compatible_slots(self, flight):
        others_slots = []
        for airline in self.airlines:
            if airline != flight.airline:
                others_slots.extend(airline.AUslots)
        return np.intersect1d(others_slots, flight.compatibleSlots, assume_unique=True)

    def offer_solution_maker(self):
        flight: IstopFlight
        airline_names = ["total"] + [airline.name for airline in self.airlines]
        flights_numbers = [self.numFlights] + [len(airline.flights) for airline in self.airlines]
        # to fix.... it / 4 works only for couples
        offers = [sum([1 for flight in self.flights if flight.slot != flight.newSlot]) / 4]
        for airline in self.airlines:
            offers.append(sum([1 for flight in airline.flights if flight.slot != flight.newSlot]) / 2)

        offers = np.array(offers).astype(int)
        self.offers = pd.DataFrame({"airline": airline_names, "flights": flights_numbers, "offers": offers})
        self.offers.sort_values(by="flights", inplace=True, ascending=False)

    @staticmethod
    def is_in(couple, couples):
        for c in couples:
            if couple[0].name == c[0].name and couple[1].name == c[1].name:
                return True
            if couple[1].name == c[0].name and couple[0].name == c[1].name:
                return True
            return False

    def f_in_matched(self, flight):
        for f in self.flights_in_matches:
            if f.name == flight.name:
                return True
        return False

    def assign_flights(self, solution_vect):
        for flight in self.flights:
            for slot in self.slots:
                if solution_vect[flight.slot.index, slot.index] > 0.9:
                    flight.newSlot = slot


"""
rows 936
problem status, explained:  mip_optimal 27612.33615924535
rows 936
sets 0
setmembers 0
elems 8140
primalinfeas 0
dualinfeas 0
simplexiter 42033
lpstatus 1
mipstatus 6
cuts 0
nodes 99
nodedepth 1
activenodes 0
mipsolnode 1042
mipsols 14
cols 2545
sparerows 793
sparecols 0
spareelems 2968
sparemipents 315
errorcode 0
mipinfeas 132
presolvestate 1310881
parentnode 0
namelength 1
qelems 0
numiis 0
mipents 2545
branchvar 0
mipthreadid 0
algorithm 2
time 1
originalrows 936
callbackcount_optnode 0
callbackcount_cutmgr 0
systemmemory 1596327
originalqelems 0
maxprobnamelength 1024
stopstatus 0
originalmipents 2545

"""
