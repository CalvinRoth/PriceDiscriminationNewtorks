from helperFun import *
import tests
import multiprocessing as mp
import pandas as pd
from stopwatch import Stopwatch
from scipy import interpolate

if __name__ == '__main__':
    a = 5
    c = 4
    rho = 0.9
    n = 700
    steps = 30
    n_trials = 15
    p0 = 1.1 / n
    p1 = np.log(n) / n
    delta = (p1 - p0) / steps
    p_space = [p0 + (delta * i) for i in range(steps)]
    """resSeq, resPar = tests.fractionalRegretvP(n, p0, p1, steps, n_trials, rho, a, c)
    plt.title("Fractional regret as a function of p")
    plt.xlabel("p")
    plt.ylabel("regret")
    plt.plot(p_space, resSeq, label="SameSeq")
    plt.plot(p_space, resPar, label="SameParam")
    plt.legend()
    plt.show()"""
    results = tests.wassersteinProfitFrac(n,p0, p1, steps, n_trials, rho, a, c)
    plt.title("Wasserstein distribution of prices")
    plt.xlabel("p")
    plt.ylabel("score")
    plt.plot(p_space, results, label="Distance")
    plt.legend()
    plt.show()
