import ctypes
from numpy.ctypeslib import ndpointer
import numpy as np
import time
from itertools import permutations
import copy
import os


class Run(object):

    def __init__(self):

        self.numProcs = os.cpu_count()
        self.lib = ctypes.CDLL('./Istop/Solvers/C_ALG/lib_run.so')
        self.lib.Run_.argtypes = [ctypes.c_void_p]
        self.lib.check_.argtypes = [ctypes.c_void_p]
        self.lib.test_.argtypes = [ctypes.c_bool]
        # self.lib.air_triple_check_.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]

        # self.couples = np.array(couples).astype(np.short)


        #         triples.remove(t)
        # self.triples = np.array(triples).astype(np.short)
        #
        # self.lib.OfferChecker_.restype = ctypes.c_void_p
        # self.obj = self.lib.OfferChecker_(ctypes.c_void_p(schedule_mat.ctypes.data),
        #                                   ctypes.c_short(schedule_mat.shape[0]),
        #                                   ctypes.c_short(schedule_mat.shape[1]),
        #                                   ctypes.c_void_p(self.couples.ctypes.data),
        #                                   ctypes.c_short(self.couples.shape[0]),
        #                                   ctypes.c_short(self.couples.shape[1]),
        #                                   ctypes.c_void_p(self.triples.ctypes.data),
        #                                   ctypes.c_short(self.triples.shape[0]),
        #                                   ctypes.c_short(self.triples.shape[1]),
        #                                   ctypes.c_short(self.numProcs))
        self.obj = self.lib.Run_(ctypes.c_void_p())
        self.lib.check_(ctypes.c_void_p(self.obj))


    def test(self, comp_matrix):
        self.lib.test_(ctypes.c_bool(comp_matrix.ctypes.data), ctypes.c_short(comp_matrix.shape[0]))

