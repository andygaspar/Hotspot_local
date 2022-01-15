import ctypes
from numpy.ctypeslib import ndpointer
import numpy as np
import time
from itertools import permutations
import copy
import os


class Run(object):

    def __init__(self, comp_matrix, reductions, offers):
        self.offers = offers
        self.solution = None
        os.system('./Istop/Solvers/C_ALG/compile.sh')
        self.numProcs = os.cpu_count()
        self.lib = ctypes.CDLL('./Istop/Solvers/C_ALG/lib_run.so')
        self.lib.Run_.argtypes = [ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
        self.lib.print_.argtypes = [ctypes.c_void_p]
        self.lib.run_.argtypes = [ctypes.c_void_p]
        self.lib.get_solution_.argtypes = [ctypes.c_void_p]
        self.lib.get_reduction_.argtypes= [ctypes.c_void_p]
        self.lib.get_solution_.restype = ndpointer(dtype=ctypes.c_bool, shape=(comp_matrix.shape[0],))
        self.lib.get_reduction_.restype = ctypes.c_double

        self.reductions = np.array(reductions)
        self.comp_matrix = comp_matrix
        self.obj = self.lib.Run_(comp_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
                                 reductions.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                 ctypes.c_size_t(comp_matrix.shape[0]))


    def test(self):
        t = time.time()
        self.lib.run_(ctypes.c_void_p(self.obj))
        t = time.time() - t
        print("c time ", t)
        sol = self.lib.get_solution_(ctypes.c_void_p(self.obj))
        self.solution = [self.offers[i].offer for i in range(len(sol)) if sol[i]]
        reduction = self.lib.get_reduction_(ctypes.c_void_p(self.obj))
        print("sol", reduction)

        for s in self.solution:
            print(s)
