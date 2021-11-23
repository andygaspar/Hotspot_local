# from mip import *
from typing import List
import numpy as np
import gurobipy

from ModelStructure.Airline.airline import Airline
from UDPP.UDPPflight import udppFlight as Fl
from ModelStructure.Slot import slot as sl
import xpress as xp
xp.controls.outputlog = 0

import ModelStructure.modelStructure as ms


class Status:

    def __init__ (self, airline, slots, x, y):
        self.bar = 1
        self.bound = 0
        self.airline = airline
        self.slots = slots
        self.x = x
        self.y = y

    def update (self):
        self.bar += 1
        print(self.bar)

#19096.41882542828

def branch_fun(problem, status:Status, obranch):
    # if problem.attributes.mipobjval - problem.attributes.bestbound < 1000:
    print("id",obranch.getid())
    if problem.attributes.bestbound > status.bound:

        status.bound = problem.attributes.bestbound
        x = []
        problem.getlpsol(x)
        y_sol = x[status.airline.numFlights ** 2:status.airline.numFlights ** 2 + status.airline.numFlights * len(
            status.slots)]
        # print("size ", len(x), len(problem.getVariable()))
        y_sol_f = np.array(y_sol).reshape((status.airline.numFlights, len(status.slots)))
        i = 0
        index = 0
        max_y = 0
        for flight in status.airline.flights:
            # print("index", flight, problem.getIndex(status.y[flight.localNum, 0]), status.airline.numFlights ** 2 + flight.localNum*len(status.slots))
            f_sol = np.argmax(y_sol_f[flight.localNum])
            if y_sol_f[flight.localNum, f_sol] > 0:
                problem.addConstraint([status.y[flight.localNum, j] == 0 for j in range(len(status.slots))])
                j_protected = f_sol
                released_slot_idx = [f.localNum for f in status.airline.flights if f.slot < status.slots[j_protected]][-1]

                cons_left = tuple([status.x[f.localNum, released_slot_idx] == 0 for
                                   f in status.airline.flights] +
                                  [status.x[flight.localNum, k] == 0 for k in range(status.airline.numFlights)] +
                                  [status.y[flight.localNum, j] == 0 if j != j_protected
                                   else status.y[flight.localNum, j] == 1
                                   for j in range(len(status.slots))])

                cons_right = ([status.y[flight.localNum, j] == 0 for j in range(len(status.slots))])

                bo = xp.branchobj(problem, branches=[cons_left, cons_right],  isoriginal=True)
                print("new_bound", problem.attributes.bestbound, " mip", problem.attributes.mipobjval,
                      "branched", problem.attributes.lpobjval)
                print("validation", bo.validate())
                bo.setpriority(1)
                return bo

def node_fun(problem, status:Status, parent, new, branch):
    print("new_bound", problem.attributes.bestbound, " mip", problem.attributes.mipobjval, "branched", problem.attributes.lpobjval, "nodes", problem.attributes.nodes, "depth", problem.attributes.nodedepth)
    # if problem.attributes.mipobjval - problem.attributes.bestbound < 1000:
    if problem.attributes.bestbound > status.bound:

        status.bound = problem.attributes.bestbound
        # x = []
        # problem.getlpsol(x)
        # y_sol = x[status.airline.numFlights ** 2:status.airline.numFlights ** 2 + status.airline.numFlights * len(
        #     status.slots)]
        # print("size ", len(x), len(problem.getVariable()))
        # y_sol_f = np.array(y_sol).reshape((status.airline.numFlights, len(status.slots)))
        # i = 0
        # index = 0
        # max_y = 0
        # for flight in status.airline.flights:
        #     # print("index", flight, problem.getIndex(status.y[flight.localNum, 0]), status.airline.numFlights ** 2 + flight.localNum*len(status.slots))
        #     f_sol = np.argmax(y_sol_f[flight.localNum])
        #     if y_sol_f[flight.localNum, f_sol] > 0:
        #         j_protected = f_sol
        #         released_slot_idx = [f.localNum for f in status.airline.flights if f.slot < status.slots[j_protected]][-1]
        #
        #         cons_left = tuple([status.x[f.localNum, released_slot_idx] == 0 for
        #                            f in status.airline.flights] +
        #                           [status.x[flight.localNum, k] == 0 for k in range(status.airline.numFlights)] +
        #                           [status.y[f.localNum, j] == 1 if f.localNum and j == j_protected
        #                            else status.y[f.localNum, j] == 1
        #                            for j in range(len(status.slots)) for f in status.airline.flights])
        #
        #         # cons_right = ([status.y[flight.localNum, j] == 0 for j in range(len(status.slots))])
        #
        #         bo = xp.branchobj(problem, branches=cons_left,  isoriginal=True)
        #         print("new_bound", problem.attributes.bestbound, " mip", problem.attributes.mipobjval,
        #               "branched", problem.attributes.lpobjval)
        #         print("validation", bo.validate(), bo.getid(), bo.getbranches())
        #         bo.setpriority(1)
        #         return bo
        # for f in y_sol:
        #     if y > 0.5:
        #         if max_y < y:
        #             max_y = y
        #             index = i
        #         # print("solution", y)
        #     i += 1
        # if max_y > 0:
        #     var = problem.getVariable()[status.airline.numFlights ** 2 + index]
        #     bo = xp.branchobj(problem, isoriginal=True)
        #     bo.addbranches(2)
        #     bo.addrows(branch=0, rowtype=['E'], rhs=[1.0], start=[0, 0], colind=[var], rowcoef=[1.0])
        #     bo.addrows(branch=1, rowtype=['E'], rhs=[0.0], start=[0, 0], colind=[var], rowcoef=[1.0])
        #     bo.setpriority(1)



            # return bo

        # print("new_bound", problem.attributes.bestbound, " mip", problem.attributes.mipobjval)

def BarrierIterCallback (problem, status):
    current_iteration = problem.attributes.bariter
    PrimalObj = problem.attributes.barprimalobj
    DualObj = problem.attributes.bardualobj
    Gap = DualObj - PrimalObj
    PrimalInf = problem.attributes.barprimalinf
    DualInf = problem.attributes.bardualinf
    ComplementaryGap = problem.attributes.barcgap
    # decide if stop or continue
    status.update()
    barrier_action = 0

    print("bound", problem.attributes.bestbound, "lp", problem.attributes.lpobjval, "mip", problem.attributes.mipobjval)
    print("nodes", problem.attributes.nodes, "depth", problem.attributes.nodedepth)
    # if (current_iteration >= 50 or
    #     Gap <= 0.1 * max (abs (PrimalObj), abs (DualObj))):
    #     print("gap", Gap)
    #     barrier_action = 2
    if problem.attributes.mipobjval - problem.attributes.bestbound < 1000:
        print("eccomi qua", status.airline.initialCosts,  status.airline.name, status.airline.numFlights)
    return barrier_action


def slot_range(k: int, AUslots: List[sl.Slot]):
    return range(AUslots[k].index + 1, AUslots[k + 1].index)


def eta_limit_slot(flight: Fl.UDPPflight, AUslots: List[sl.Slot]):
    i = 0
    for slot in AUslots:
        if slot >= flight.etaSlot:
            return i
        i += 1


def get_num_flights_for_eta(flight: Fl.UDPPflight, airline: Airline):
    return sum([1 for fl in airline.flights if fl.etaSlot.time == flight.etaSlot.time])


def UDPPlocalOptXp(airline: Airline, slots: List[sl.Slot]):

    m = xp.problem()
    xp.controls.outputlog = 1
    x = np.array([[xp.var(vartype=xp.binary) for _ in airline.flights] for _ in airline.flights])

    z = np.array([xp.var(vartype=xp.integer) for _ in airline.flights])

    y = np.array([[xp.var(vartype=xp.binary) for _ in slots] for _ in airline.flights])

    m.addVariable(x, y, z)

    flight: Fl.UDPPflight

    m.addConstraint(
        xp.Sum(x[0, k] for k in range(airline.numFlights)) == 1
    )


    for flight in airline.flights:
        for j in range(len(slots)):
            for k in range(airline.numFlights):
                m.addConstraint(
                    x[flight.localNum, k] <= 1 - y[flight.localNum, j]
                )

                m.addConstraint(
                    1 - x[flight.localNum, k] >= y[flight.localNum, j]
                )

    # slot constraint
    for j in slots:
        #one y max for slot
        m.addConstraint(
            xp.Sum(y[flight.localNum, j.index] for flight in airline.flights) <= 1
        )

    for flight in airline.flights:

        # m.addConstraint(
        #     [y[flight.localNum, j] == 0 for j in range(flight.etaSlot.index + airline.numFlights - flight.localNum, len(slots))]
        # )

        eta_index = flight.etaSlot.index
        end_index = eta_index + get_num_flights_for_eta(flight, airline)
        m.addConstraint(
            [y[flight.localNum, slot.index] == 0 for slot in slots if slot.index not in range(eta_index, end_index)]
        )


    for k in range(airline.numFlights - 1):
        #one x max for slot
        m.addConstraint(
            xp.Sum(x[flight.localNum, k] for flight in airline.flights) <= 1
        )


        m.addConstraint(
            [y[flight.localNum, airline.AUslots[k].index] == 0 for flight in airline.flights]
        )

        m.addConstraint(
            xp.Sum(y[i, j] for i in range(k, airline.numFlights) for j in range(airline.AUslots[k].index)) <= \
             xp.Sum(x[i, kk] for i in range(k + 1) for kk in range(k, airline.numFlights))
        )



        m.addConstraint(
            xp.Sum(y[flight.localNum, j] for flight in airline.flights for j in slot_range(k, airline.AUslots)) \
             == z[k]
        )

        m.addConstraint(
            xp.Sum(y[flight.localNum, j] for flight in airline.flights for j in range(airline.AUslots[k].index)) <= \
             xp.Sum(x[i, j] for i in range(k) for j in range(k, airline.numFlights))
        )

        for i in range(k + 1):
            m.addConstraint(
                (1 - xp.Sum(x[flight.localNum, i] for flight in airline.flights)) * 1000 \
                 >= z[k] - (k - i)
            )
    # last slot
    m.addConstraint(
        xp.Sum(x[flight.localNum, airline.numFlights - 1] for flight in airline.flights) == 1
    )

    for flight in airline.flights:
        m.addConstraint(
            [y[flight.localNum, j] == 0 for j in range(flight.etaSlot.index)]
        )

    for flight in airline.flights[1:]:
        # flight assignment
        m.addConstraint(
            xp.Sum(y[flight.localNum, j] for j in range(flight.etaSlot.index, flight.slot.index)) + \
            xp.Sum(x[flight.localNum, k] for k in
                  range(eta_limit_slot(flight, airline.AUslots), airline.numFlights)) == 1
        )

    # not earlier than its first flight
    m.addConstraint(
        [y[flight.localNum, j] == 0 for flight in airline.flights for j in range(airline.flights[0].slot.index)]
    )

    m.setObjective(
            xp.Sum(y[flight.localNum][slot.index] * flight.cost_fun(slot)
             for flight in airline.flights for slot in slots) +
            xp.Sum(x[flight.localNum][k] * flight.cost_fun(airline.AUslots[k])
             for flight in airline.flights for k in range(airline.numFlights))
    )

    # solval = np.concatenate([pre_solve(airline), np.zeros(y.shape[0]*y.shape[1] + z.shape[0])])
    solval, cost_ = pre_solve(airline)
    print("inital bound", cost_)
    solval = np.concatenate([solval, np.zeros(y.shape[0] * y.shape[1] + z.shape[0])])
    # m.loadmipsol(solval)
    # m.addmipsol(np.zeros(y.shape[0] * y.shape[1]), y.flatten())
    # m.addmipsol(np.zeros(z.shape[0]), z)

    status = Status(airline, slots, x, y)
    m.addcbnewnode(node_fun, status, 0)
    m.setControl({'maxtime': 2})
    # m.addcbintsol(BarrierIterCallback, status, 0)
    # m.addcbchgbranchobject(branch_fun, status, 0)
    m.solve()
    print(m.getProbStatusString())
    # print("airline ",airline)
    n_flights = []
    for flight in airline.flights:

        f_sol = np.argwhere(m.getSolution(y[flight.localNum]) > 0.5)
        if f_sol.shape[0] > 0:
            flight.newSlot = slots[f_sol[0, 0]]
            flight.udppPriority = "P"
            flight.tna = slots[f_sol[0, 0]].time
            airline.protections += 1
        else:
            f_sol = np.argwhere(m.getSolution(x[flight.localNum]) > 0.5)
            flight.newSlot = airline.flights[f_sol[0, 0]].slot
            flight.udppPriority = "N"
            flight.udppPriorityNumber = f_sol[0, 0]
            n_flights.append(flight)

    # n_flights.sort(key=lambda f: f.udppPriorityNumber)
    # for i in range(len(n_flights)):
    #     n_flights[i].udppPriorityNumber = i
    fl = airline.flights
    fl.sort(key=lambda f: f.newSlot.time)
    sol=[(f.name, f.newSlot.index, f.etaSlot.index) for f in fl]
    print("done")

    return sol

def pre_solve(airline: Airline):

    m = xp.problem()

    x = np.array([[xp.var(vartype=xp.binary) for _ in airline.flights] for _ in airline.flights])

    m.addVariable(x)

    flight: Fl.UDPPflight


    # slot constraint
    for j in range(airline.numFlights):
        # one y max for slot
        m.addConstraint(
            xp.Sum(x[flight.localNum, j] for flight in airline.flights) <= 1
        )

    for flight in airline.flights:
        # flight assignment
        m.addConstraint(
            xp.Sum(x[flight.localNum, k] for k in
                   range(eta_limit_slot(flight, airline.AUslots), airline.numFlights)) == 1
        )

        m.addConstraint(
            [x[flight.localNum, k] == 0 for k in range(eta_limit_slot(flight, airline.AUslots))]
        )


    m.setObjective(
        xp.Sum(x[flight.localNum][k] * flight.cost_fun(airline.AUslots[k])
               for flight in airline.flights for k in range(airline.numFlights))
    )

    m.solve()

    return m.getSolution(x).flatten(), m.getObjVal()
