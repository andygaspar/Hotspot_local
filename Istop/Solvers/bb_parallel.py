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
    if len(node.offers) == 0:
        return State(node.reduction, node.solution, True), []
    child_list = []
    new_state = state

    l_reduction = node.reduction + node.offers[0].reduction
    l_solution = node.solution + [node.offers[0]]

    if l_reduction > state.reduction:
        new_state = State(l_reduction, l_solution, state.init_solution)

    l_incompatible = [offer for flight in node.offers[0].flights for offer in flight.offers]
    l_offers = [offer for offer in node.offers[1:] if offer not in l_incompatible]

    pruned = False
    if state.init_solution:
        l_offers_reduction = sum([offer.reduction for offer in l_offers])
        bound = l_reduction + l_offers_reduction
        if bound < state.reduction:
            pruned = True

    if not pruned:
        child_list.append(Node(l_reduction, l_solution, l_offers))

    r_offers = node.offers[1:]

    pruned = False

    r_offers_reduction = sum([offer.reduction for offer in r_offers])
    bound = node.reduction + r_offers_reduction
    if bound < state.reduction:
        pruned = True

    if not pruned:
        child_list.append(Node(node.reduction, node.solution, r_offers))

    return new_state, child_list

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