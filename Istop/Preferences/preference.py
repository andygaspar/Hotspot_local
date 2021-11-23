from matplotlib import pyplot as plt
from scipy.optimize import curve_fit, minimize
import scipy
import math
import numpy as np
import time
import multiprocessing
from multiprocessing import Pool

num_cpu = multiprocessing.cpu_count()


def approx_linear(x, slope):
    return slope * x


def approx_one_margins(x, margin_1, jump_1):
    cost = np.zeros_like(x)
    cost[margin_1 <= x] = jump_1
    return cost


def approx_two_margins(x, margin_1, jump_1, margin_2, jump_2):
    cost = approx_one_margins(x, margin_1, jump_1)
    cost[margin_2 <= x] = jump_2
    return cost


def approx_slope_one_margin(x, slope, margin_1, jump_1):
    cost = approx_linear(x, slope)
    cost[margin_1 <= x] = jump_1
    return cost


def approx_slope_two_margins(x, slope, margin_1, jump_1, margin_2, jump_2):
    cost = approx_slope_one_margin(x, slope, margin_1, jump_1)
    cost[margin_2 <= x] = jump_2
    return cost


def approx_three_margins(x, slope, margin_1, jump_1, margin_2, jump_2, margin_3, jump_3):
    cost = approx_slope_two_margins(x, slope, margin_1, jump_1, margin_2, jump_2)
    cost[margin_3 <= x] = jump_3
    return cost


def loss_fun(Y_test, y_train):
    return sum(
        [(y_train[i] - Y_test[i]) ** 2 /10 if y_train[i] - Y_test[i] > 0 else (Y_test[i] - y_train[i]) ** 2 for i in
         range(Y_test.shape[0])])


def obj_approx(params, x, y, approx_fun):

    return loss_fun(y, approx_fun(x, *params))


def fit_curve(vals):
    x, y = vals[:2]
    approx_fun = vals[-1]
    params = vals[2:-1]
    return loss_fun(y, approx_fun(x, *params))


def fit_cost_curve(x, y, max_delay, steps=8):
    test_values = []
    max_val = max(y)
    for slope in np.linspace(0, 1, steps):
        for margin_1 in np.linspace(0, 3*max_delay//4, steps):
            for jump_1 in np.linspace(10, max_val, steps//2):
                for margin_2 in np.linspace(margin_1, max_delay, steps//2):
                    for jump_2 in np.linspace(jump_1, max_val, steps//2):
                        test_values.append(
                            (x, y, slope, margin_1, jump_1, margin_2, jump_2, approx_slope_two_margins))
    pool = Pool(num_cpu)
    guesses = pool.map(fit_curve, test_values)
    best_initial_guess = np.array(test_values[np.argmin(guesses)][2:-1])

    solution = minimize(obj_approx, best_initial_guess, args=(x, y, approx_slope_two_margins),
                        method='Powell', options={'maxiter': 10000, 'xtol': 0.5, 'ftol': 0.01})
    return solution.x


def make_preference_fun(max_delay: float, delay_cost_vect: np.array):
    delays = np.linspace(0, max_delay + 0.1, delay_cost_vect.shape[0])
    result = fit_cost_curve(delays, delay_cost_vect, max_delay)
    slope, margin_1, jump_1, margin_2, jump_2 = result
    # plt.plot(delay_cost_vect)
    # plt.plot(approx_slope_two_margins(delays, slope, margin_1, jump_1, margin_2, jump_2))
    # plt.show()
    return slope, margin_1, jump_1, margin_2, jump_2

