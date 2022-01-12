import sys
import time
from typing import List

import numpy as np
import matplotlib
from _distutils_hack import override
from matplotlib import pyplot as plt

from Istop.AirlineAndFlight.istopFlight import IstopFlight
from gurobipy import Model, GRB, quicksum, Env
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout

from Istop.Solvers.bb import BB, Offer
from Istop.Solvers import bb
from Istop.old.bb_old import get_offers_for_flight

from collections import namedtuple

Node = namedtuple('Node', ['reduction', 'solution', 'offers'])
State = namedtuple('State', ['reduction', 'solution', 'init_solution'])

init_state = State(0, 0, False)


def update_state(new_state, old_state):
    return new_state if new_state.reduction > old_state.reduction else old_state


def process_node(node: Node, state: State):
    self.nodes += 1
    if np.sum(offers) == 0:
        self.initSolution = True
        return 0

    idx = np.nonzero(offers)[0][0]

    l_reduction = reduction + reductions[idx]
    l_solution = copy.copy(solution)
    l_solution[idx] = True

    if l_reduction > self.best_reduction:
        self.solution = l_solution
        self.best_reduction = l_reduction

    l_offers = comp_matrix[idx] * offers
    l_offers[idx] = False

    l_offers_key = np.nonzero(l_offers)[0].tobytes()

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
            l_offers_reduction = sum(reductions * l_offers)
            bound = l_reduction + l_offers_reduction

            if bound < self.best_reduction:
                pruned = True
                best_left = l_offers_reduction

    if not pruned:
        best_left = self.step(l_solution, l_offers, l_reduction, reductions, comp_matrix)

    r_offers = offers
    r_offers[idx] = False
    r_offers_key = np.nonzero(r_offers)[0].tobytes()

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
        r_offers_reduction = sum(reductions * r_offers)
        bound = reduction + r_offers_reduction
        if bound < self.best_reduction:
            pruned = True
            best_right = r_offers_reduction

    if not pruned:
        best_right = self.step(solution, r_offers, reduction, reductions, comp_matrix)

    best = max(best_left + reductions[idx], best_right)
    r_offers[idx] = True
    key = np.nonzero(r_offers)[0].tobytes()
    self.precomputed[key] = best

    return best

def pruning_condition(node, state):
    return False


def multinode_processing(node: Node, state: State):
    child_list = [node]
    new_state = state

    node_count = 0
    max_node = 5_000 if node.solution else 1000

    while node_count < max_node and len(child_list) > 0:

        current_node = child_list.pop(-1)

        if len(current_node.offers) == 0:
            new_state = State(current_node.reduction, current_node.solution, True) \
                if new_state.reduction < current_node.reduction else new_state
            continue

        r_offers = current_node.offers[1:]

        r_offers_reduction = sum([offer.reduction for offer in r_offers])
        bound = current_node.reduction + r_offers_reduction
        if not bound < state.reduction:
            child_list.append(Node(current_node.reduction, current_node.solution, r_offers))

        l_reduction = current_node.reduction + current_node.offers[0].reduction
        l_solution = current_node.solution + [current_node.offers[0]]

        if l_reduction > new_state.reduction:
            new_state = State(l_reduction, l_solution, state.init_solution)

        l_incompatible = [offer for flight in current_node.offers[0].flights for offer in flight.offers]
        l_offers = [offer for offer in current_node.offers[1:] if offer not in l_incompatible]

        if state.init_solution:
            l_offers_reduction = sum([offer.reduction for offer in l_offers])
            bound = l_reduction + l_offers_reduction
            if not bound < state.reduction:
                child_list.append(Node(l_reduction, l_solution, l_offers))
        else:
            child_list.append(Node(l_reduction, l_solution, l_offers))

        node_count += 1

    print(new_state, child_list)
    return new_state, child_list