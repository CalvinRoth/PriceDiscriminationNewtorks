from __future__ import annotations

import numpy as np
import numpy.linalg as lin
import networkx as nx
import matplotlib.pyplot as plt


# Linear algebra
def specNorm(A: np.matrix) -> float:
    return lin.norm(A, ord=2)
    # return np.sqrt(slin.eigs(A.T @ A, k=1, which="LM", return_eigenvectors=False, tol=1e-10)[0])


# Graph makers
def makeSimilarGraph(G: nx.DiGraph) -> np.matrix:
    """ Generates the new graph with the same in/out degree as the orginal
        -------
        Return adj. matrix of graph
    """
    sequence_in = [d for _, d in G.in_degree]
    sequence_out = [d for _, d in G.out_degree]
    return nx.to_numpy_matrix(
        nx.directed_configuration_model(sequence_in, sequence_out, create_using=nx.DiGraph),
        dtype="d"
    )


# Graph Generators
def makeERGraph(n: int, p: float) -> np.matrix:
    """ Generates Random Erdos-Renyi Graphs with n vertices and link probability p
        ------
        return Adjacency graph of matrix and the networkx DiGraph object
    """
    G = nx.generators.fast_gnp_random_graph(n, p, directed=True)
    # sortG = sorted(G.in_degree, key=lambda x: x[1], reverse=True)
    return nx.to_numpy_matrix(G, dtype="d"), G


def centralty(A: np.matrix, rho: float) -> np.matrix:
    """

    Parameters
    ----------
    A : np matrix
    rho : network effect

    Returns
    -------
    Centrality vector as described in paper
    """
    n = A.shape[0]

    ident = np.eye(n, n)
    ones = np.ones((n, 1))
    ApA = A + A.T
    eig = specNorm(ApA)
    alpha = rho / eig
    central = lin.inv(ident - (alpha * ApA))
    central = central @ ones  # Checked.  this > 0
    return central


# Paper related properties
def applyPriceVector(A: np.matrix, v: np.matrix, rho: float, a: int | float, c: int | float) -> (float, bool) :
    """

    Parameters
    ----------
    A : Graph
    v : price vector
    rho : network strength
    a : Stand alone strength
    c : Marginal cost. Should be less than a

    Returns
    -------
    Profit in this network if prces v were applied.
    And if result is valid or not
    """
    n = A.shape[0]
    ident = np.eye(n, n)
    ones = np.ones((n, 1))
    ApA = A + A.T
    spN = specNorm(ApA)  # Sometimes scipy return x+0i, this is to discard warning
    consumption = (2 * rho / spN) * A
    consumption = ident - consumption
    consumption = 0.5 * lin.inv(consumption)  # This is entirely in the range [0,1] ^ checked

    consumption = consumption @ ((a * ones) - v)
    valid = True
    if(np.min(consumption) < 0):
        valid = False
    return ((v - (c * ones)).T @ consumption)[0, 0], valid


def priceVector(A: np.matrix, rho: float, a: int | float, c: int | float) -> np.matrix:
    """
    Parameters
    ----------
    A : Network
    rho : network strength
    a : stand alone util
    c : marginal cost. Should be less than a

    Returns
    -------
    Vector reprsenting what price to charge individual i
    """
    n = A.shape[0]
    ones = np.ones((n, 1))
    alpha = rho / specNorm(A + A.T)
    central = centralty(A, rho)  # This should be A not A + A.T because of how centralty function is designed
    dif = A - A.T
    pv1 = ((a + c) / 2) * ones
    pv2 = ((a - c) * alpha * 0.5) * (dif @ central)
    return pv1 + pv2


def optimalProfit(A: np.matrix, n: int, a: int | float, c: int | float, rho: float):
    """
    Parameters
    ----------
    A : Network
    n : size of network
    rho : network strength
    a : stand alone util
    c : marginal cost. Should be less than a
    Returns
    -------
    True profit. Should be the same as applyPriceVector(A, pricevector(A,...),...)
    """
    one = np.ones((n, 1))
    weight = rho / specNorm(A + A.T)
    t1 = lin.inv(np.eye(n, n) - (weight * (A + A.T)))
    total = one.T @ t1 @ one
    total = ((a - c) * (a - c) / 8) * total
    return np.real(total[0, 0])


def fractionalRegret(A, v, n, rho, a, c):
    """

    Parameters
    ----------
    A : Network
    v : price vector to compare to
    rho : network strength
    n : number of nodes
    a : stand alone util
    c : marginal cost. Should be less than a

    Returns
    -------
    1 - (profit of A using v)/(profit of A using best choice)
    """
    discrim = optimalProfit(A, n, a, c, rho)  # Optimal profit
    # I have check and the formula for optimal profit does match applypricevector(A, pricevector(A,...), params)
    appliedProf = applyPriceVector(A, v, rho, a, c)  # Profit at v
    return 1 - (appliedProf / discrim)

def genGoodSeqProfit(n :int , G : nx.digraph, A : np.matrix, rho : float, a : int|float, c : int|float, alpha):
    i = 0
    while True:
        A_seq = makeSimilarGraph(G)  # for same seq
        v_seq = priceVector(A_seq, rho, a, c)
        (profit, flag) = applyPriceVector(A, v_seq, rho, a, c)
        if (flag):
            break
        i += 1
    print("Seq", i)
    return v_seq, profit


def genGoodParamProfit(n :int , G : nx.digraph, A : np.matrix, rho : float, a : int|float, c : int|float, alpha):
    i = 0
    while True:
        A_par, B = makeERGraph(n, p)  # Same Param
        v_par = priceVector(A_par, rho, a, c)
        (profit, flag) = applyPriceVector(A, v_par, rho, a, c)
        if (flag):
            break
        i += 1
    print("param", i)
    return v_par, profit

def fractionalRegretvP(n: int, p0: float, p1: float, steps: int, n_trials: int, rho: float, a: int | float,
                       c: int | float) -> (np.matrix, np.matrix):
    """
    Parameters
    ----------
    n : Size of Graphs
    p0 : Lower bound of prob. of link formation
    p1 : Upper bound of prob of link formation
    steps : How many different Ps to sample
    n_trials : How many trials to run at that p
    rho : Network effect
    a : Standalone effect
    c : Cost

    Returns
    -------
    Returns 2 vectors: One sequence of resutls for same sequence graphs and one for same parameter graphs.
    """
    results_seq = np.zeros(steps)
    results_param = np.zeros(steps)
    delta = (p1 - p0) / steps
    Ps = [p0 + (i * delta) for i in range(steps)]
    alpha = 0.1
    for i, p in enumerate(Ps):
        A_true, G_true = makeERGraph(n, p)
        A_test = np.copy(A_true)
        seqScore = 0
        parScore = 0
        v = np.zeros((n, 1))
        true_profit = optimalProfit(A_true, n, a, c, rho)
        for j in range(n_trials):
            v_par, profit_par = genGoodParamProfit(n, G_true, A_true, rho, a, c, alpha)
            v_seq, profit_seq = genGoodSeqProfit(n, G_true, A_true, rho, a, c)
            seqScore = 1 - (profit_seq / true_profit)
            parScore = 1 - (profit_par / true_profit)
        parScore /= n_trials
        seqScore /= n_trials
        results_seq[i] = seqScore
        results_param[i] = parScore
    return results_seq, results_param


rho = 0.9
a = 5
c = 4
#  Small n for speed
n = 1000
n_trials = 30
steps = 30
# counting = [i for i in range(n)]
p = np.log(n) / n
p0 = 1.1 / n
p1 = np.log(n) / n
delta = (p1 - p0) / steps
p_space = [p0 + (delta * i) for i in range(steps)]
resSeq, resPar = fractionalRegretvP(n, p0, p1, n_trials, steps, rho, a, c)
""" Uncomment me for graph"""
plt.title("Fractional regret as a function of p")
plt.xlabel("p")
plt.ylabel("regret")
plt.plot(p_space, resSeq, label="SameSeq")
plt.plot(p_space, resPar, label="SameParam")
plt.legend()
plt.show()

## Graphs in paper
a = 5
c = 4
rho = 0.9
#  A
A = np.zeros((5, 5))
for i in range(1, 5):
    A[0, i] = 1
    A[i, 0] = 1
Av = priceVector(A, rho, a, c)

#  B
B = np.zeros((3, 3))
B[0, 1] = 1
B[1, 2] = 1
B[2, 0] = 1
Bv = priceVector(B, rho, a, c)

# C
C = np.zeros((4, 4))
C[0, 1] = 1
C[0, 3] = 1
C[1, 2] = 1
C[2, 0] = 1
C[3, 0] = 1
Cv = priceVector(C, rho, a, c)

# D
D = np.zeros((4, 4))
D[0, 1] = 1
D[0, 3] = 1
D[1, 0] = 1
D[1, 2] = 1
D[2, 1] = 1
D[2, 3] = 1
D[3, 1] = 1
D[3, 2] = 1
Dv = priceVector(D, rho, a, c)

# E
E = np.zeros((3, 3))
E[0, 1] = 1
E[2, 0] = 1
E[2, 1] = 1
Ev = priceVector(E, rho, a, c)
