import numpy as np
import numpy.linalg as lin
import networkx as nx
import scipy
import scipy.sparse.linalg as slin
import matplotlib.pyplot as plt


# #  Preliminaries
# First we implement the powerIteration method in order to quickly estimate the largest eigenvalue

def powerIteration(A, iters=20):
    n = A.shape[0]
    b = np.random.randn(n, )
    b = b / lin.norm(b, ord=2)
    for i in range(iters):
        bnew = A @ b
        b = bnew / lin.norm(bnew, ord=2)
    return b


def snorm(A):
    b = powerIteration(A.T @ A)
    return np.sqrt((b.T @ A.T @ A @ b))


def specNorm(A):
    return np.sqrt(slin.eigs(A.T @ A, k=1, which="LM", return_eigenvectors=False, tol=1e-5)[0])


# Graph Generators
def makeERGraph(n, p):
    """ Generates Random Erdos-Renyi Graphs with n vertices and link probability p
        ------
        return Adjacency graph of matrix and the networkx DiGraph object
    """
    G = nx.generators.fast_gnp_random_graph(n, p, directed=True)
    # sortG = sorted(G.in_degree, key=lambda x: x[1], reverse=True)
    return nx.to_scipy_sparse_matrix(G, dtype="d"), G


def makeSimilarGraph(G):
    """ Generates the new graph with the same in/out degree as the orginal
        -------
        Return adj. matrix of graph
    """
    sequence_in = [d for n, d in G.in_degree()]
    sequence_out = [d for n, d in G.out_degree()]
    return nx.to_scipy_sparse_matrix(
        nx.directed_configuration_model(sequence_in, sequence_out, create_using=nx.DiGraph),
        dtype="d"
    )


# Profits 
def profitUniform(A, rho, a, c):
    n = A.shape[0]
    ones = np.ones((n, 1))
    alpha = 2 * rho / specNorm(A + A.T)
    temp = np.eye(n, n) - alpha * A
    temp = lin.inv(temp)
    return (((a - c) * (a - c)) / 8) * (ones.T @ temp @ ones)[0, 0]


# Optimal profit under price discrim
def profitDiscrim(A, n, p, a, c, rho):
    # A and C should only impact graph up to scaling
    one = np.ones((n, 1))
    weight = rho / specNorm(A + A.T)
    t1 = lin.inv(np.eye(n, n) - (weight * (A + A.T)))
    total = one.T @ t1 @ one
    total = ((a - c) * (a - c) / 8) * total
    return np.real(total[0, 0])


# Price Vectors
def priceVector(A, rho, a, c):
    """ Get the price vector of graph A """
    n = A.shape[0]

    ident = np.eye(n, n)
    ones = np.ones((n, 1))
    ApA = A + A.T
    eig = np.real(specNorm(ApA))
    alpha = rho / eig
    central = lin.inv(ident - alpha * ApA)
    central = central @ ones
    dif = A - A.T
    return ((a + c) / 2) * ones + ((a - c) * alpha / 2) * (dif @ central)


def applyPriceVector(A, v, rho, a, c):
    """ Apply price vector to find the profits of A"""
    n = A.shape[0]
    ident = np.eye(n, n)
    ones = np.ones((n, 1))
    ApA = A + A.T
    spN = np.real(specNorm(ApA))  # Sometimes scipy return x+0i, this is to discard warning
    consumption = (2 * rho / spN) * A
    consumption = ident - consumption;
    consumption = 0.5 * lin.inv(consumption)
    consumption = consumption @ ((a * ones) - v)
    #if (np.min(consumption) < 0):
        # My understanding is this should literally never print
     #   print(spN, np.min(v), np.min(A))
    return (v - c * ones).T @ consumption


# Gaps applying price vector of G to guesses
def applyTrueVector(A, test, rho, a, c):
    """ Apply optimal profit price vector A to test graph test and A. Return pair of profits"""
    optimalVector = priceVector(A, rho, a, c)
    profitAtGuess = applyPriceVector(test, optimalVector, rho, a, c)
    trueProfit = applyPriceVector(A, optimalVector, rho, a, c)
    return trueProfit, profitAtGuess


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


def fractionalGap(n, p, rho, a, c, i, results, n_trials):
    A, G = makeERGraph(n, p)
    n = A.shape[0]
    trueProfit = np.real(applyPriceVector(A, priceVector(A, rho, a, c), rho, a, c))
    # the average vector initilized with sample size of 1
    averageV = priceVector(makeSimilarGraph(G), rho, a, c)
    # And another n_trials-1 trials
    for j in range(n_trials - 1):
        averageV += priceVector(makeSimilarGraph(G), rho, a, c)
    averageV /= n_trials  # Scaling
    profit = np.real(applyPriceVector(A, averageV, rho, a, c))
    results[i] = trueProfit / profit


# Get the variance of the price vectors
def varianceVector(n, p, rho, a, c, n_trials):
    A, G = makeERGraph(n, p)
    generated_vs = np.zeros((n, n_trials))
    true_vector = priceVector(A, rho, a, c)
    for i in range(n_trials):
        v = true_vector - priceVector(makeSimilarGraph(G), rho, a, c)
        v.resize((n,))
        generated_vs[:, i] = v
    return np.var(generated_vs, axis=1)


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


def fractionalRegret(A, v, n, p, rho, a, c):
    """ Fractional regret relative to price vector v """
    discrim = profitDiscrim(A, n, p, a, c, rho)  # Optimal profit
    ds2 = applyPriceVector(A, priceVector(A,rho,a,c),rho, a, c)
    if(np.abs(discrim - ds2) > 0.00001 ):
        print("Hmmmm")
    appliedProf = np.real(applyPriceVector(A, v, rho, a, c))  # Profit at v
    return 1 - (appliedProf / discrim)
