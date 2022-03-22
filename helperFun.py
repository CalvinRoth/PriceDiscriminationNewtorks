from __future__ import annotations
import numpy as np
import numpy.linalg as lin
import networkx as nx
import scipy
import scipy.sparse.linalg as slin
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


def centralty(A: np.matrix, rho: float, alpha) -> np.matrix:
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
def applyPriceVector(A: np.matrix, v: np.matrix, rho: float, a: int | float, c: int | float) -> (float, bool):
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
    # spN = specNorm(ApA)  # Sometimes scipy return x+0i, this is to discard warning
    alpha = (rho / specNorm(ApA))
    consumption = (2 * alpha) * A
    consumption = ident - consumption
    consumption = 0.5 * lin.inv(consumption)  # This is entirely in the range [0,1] ^ checked

    consumption = consumption @ ((a * ones) - v)
    valid = True
    if (np.min(consumption) < 0):
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
    central = centralty(A, rho, alpha)  # This should be A not A + A.T because of how centralty function is designed
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
    alpha = rho / specNorm(A + A.T)

    t1 = lin.inv(np.eye(n, n) - (alpha * (A + A.T)))
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


# Checks to regen price vector

def genGoodSeqProfit(n: int, G: nx.digraph, A: np.matrix, rho: float, a: int | float, c: int | float):
    i = 0
    flag = True
    v_seq = 0
    profit = 0
    while flag:
        A_seq = makeSimilarGraph(G)  # for same seq
        v_seq = priceVector(A_seq, rho, a, c)
        (profit, flag) = applyPriceVector(A, v_seq, rho, a, c)
        i += 1
        if i >= 1:
            flag = False
    return v_seq, profit


def genGoodParamProfit(n: int, p : float, G: nx.digraph, A: np.matrix, rho: float, a: int | float, c: int | float):
    i = 0
    flag = True
    v_par = 0
    profit = 0
    while flag:
        A_par, B = makeERGraph(n, p)  # Same Param
        v_par = priceVector(A_par, rho, a, c)
        (profit, flag) = applyPriceVector(A, v_par, rho, a, c)
        i += 1
        if i >= 1:
            flag = False
    return v_par, profit


# Gaps applying price vector of guesses to true graph G
def getGapRev(A, test, rho, a, c):
    """ Apply optimal profit price vector guess graph to test graph test and A. Return pair of profits"""
    optimalVector = priceVector(test, rho, a, c)
    profitWithGuessV = applyPriceVector(A, optimalVector, rho, a, c)
    trueProfit = applyPriceVector(A, priceVector(A, rho, a, c), rho, a, c)
    return trueProfit, profitWithGuessV


# Applying the true optimal profit vector to guesses
# Currently not using because it seems backwards of what I want
def getGaps(n, p, rho, a, c, i, results, n_trials):
    A, G = makeERGraph(n, p)
    results[i] = np.average([getGap(A, makeSimilarGraph(G), rho, a, c) for j in range(n_trials)])
    return i


# Apply the price vector that each guess produces to the true graph and take average.
def getGapsReverse(n, p, rho, a, c, i, results, n_trials):
    A, G = makeERGraph(n, p)
    results[i] = np.average([getGapRev(A, makeSimilarGraph(G), rho, a, c) for j in range(n_trials)])
    return i


# Here we apply the average optimal price vector of the guesses to the true graph G
# I get a warning about discarding complex values, values should never complex for this problem
# and when I check they are all +0i so ?


def getAverageGap(n, p, rho, a, c, i, results, n_trials):
    A, G = makeERGraph(n, p)
    n = A.shape[0]
    trueProfit = applyPriceVector(A, priceVector(A, rho, a, c), rho, a, c)
    # the average vector initilized with sample size of 1
    averageV = priceVector(makeSimilarGraph(G), rho, a, c)
    # And another n_trials-1 trials
    for j in range(n_trials - 1):
        averageV += priceVector(makeSimilarGraph(G), rho, a, c)
    averageV /= n_trials  # Scaling
    profit = applyPriceVector(A, averageV, rho, a, c)
    results[i] = np.real(trueProfit - profit)


# How much does the profit change when we change the ith coordinate of price vector
# Change each percent wise. Test +/- percent
def robustNess(n, p, chaos, rho, a, c):
    A, G = makeERGraph(n, p)
    true_vector = priceVector(A, rho, a, c)
    true_profit = applyPriceVector(A, true_vector, rho, a, c)
    results = np.zeros(n)
    count_range = [i for i in range(n)]
    inD = [d[1] for d in G.in_degree]
    outD = [d[1] for d in G.out_degree]

    plt.plot(inD)
    plt.plot(outD)
    plt.show()
    """for i in range(n):
        increase_v = true_vector
        decrease_v = true_vector
        increase_v[i] += chaos * increase_v[i];
        decrease_v[i] -= chaos * decrease_v[i];
        profitI = true_profit - applyPriceVector(A, increase_v, rho, a,c);
        profitD = true_profit - applyPriceVector(A, decrease_v, rho, a,c);
        results[i] = max(profitI, profitD)"""
    return results
