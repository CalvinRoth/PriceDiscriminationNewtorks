import numpy as np
import numpy.linalg as lin
import networkx as nx
import scipy
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency 



# #  Prelimaries 
# First we implement the powerIteration method in order to quickly estimate the largest eigenvalue

def powerIteration(A, iters=20):
    n = A.shape[0]
    b = np.random.randn(n,)
    b = b / lin.norm(b, ord=2)
    for i in range(iters):
        bnew = A @ b
        b = bnew / lin.norm(bnew, ord = 2)
    return b

def snorm(A):
    b = powerIteration(A.T @ A)
    return np.sqrt((b.T @ A.T @ A @ b))

def specNorm(A):
    return  np.sqrt(scipy.sparse.linalg.eigs(A.T @ A, k=1, which="LM",return_eigenvectors=False, tol=1e-5)[0])


#Graph Generators
def makeERGraph(n,p):
    """ Generates Random Erdos-Renyi Graphs with n vertices and link probability p
        Returns 
        ------
        Adjacency graph of matrix and the networkx DiGraph object
    """
    G = nx.generators.fast_gnp_random_graph(n,p, directed=True)
    return nx.to_scipy_sparse_matrix(G, dtype="d"), G


def makeSimilarGraph(G):

    sequence_in = [d for n,d in G.in_degree()]
    sequence_out = [d for n,d in G.out_degree()]
    return nx.to_scipy_sparse_matrix(
        nx.directed_configuration_model(sequence_in, sequence_out, create_using=nx.DiGraph),
        dtype="d"
    )



# Profits 
def profitUniform(A, rho):
    n = A.shape[0]
    alpha = 2*rho/specNorm(A + A.T)
    temp = I - alpha* A
    temp = lin.inv(temp)
    return (ones.T @ temp @ ones)[0,0]


def profitDiscrim(A,rho):
    n = A.shape[0]
    alpha = rho / specNorm(A + A.T)
    inner = I - alpha*(A + A.T)
    return (ones.T @ lin.inv(inner) @ ones)[0,0]


# Price Vectors
def priceVector(A, rho,a,c ):
    n = A.shape[0]
    
    ident = np.eye(n,n)
    ones = np.ones((n,1))
    ApA = A + A.T
    eig = specNorm(ApA)
    alpha = rho / eig
    central = lin.inv(ident - alpha*ApA) @ ones
    dif = A - A.T
    return ((a+c)/2)*ones +  ((a-c)*alpha/2) * dif @ central


def applyPriceVector(A, v, rho,  a, c):
    n = A.shape[0]
    ident = np.eye(n,n)
    ones = np.ones((n,1))
    ApA = A + A.T
    consumption = (2*rho/specNorm(ApA)) * A
    consumption = 0.5 * lin.inv(ident - consumption)
    consumption = consumption @ (a*ones - v)
    return ((v - c*ones).T @ consumption)[0,0]



# Gaps applying price vector of G to guesses
def getGap(A, test, rho, a, c):
    optimalVector = priceVector(A, rho, a, c)
    profitAtGuess = applyPriceVector(test, optimalVector, rho, a,c)
    trueProfit = applyPriceVector(A, optimalVector, rho, a,c)
    return trueProfit - profitAtGuess



# Gaps applying price vector of guesses to G
def averageGapReverse(A,test, rho, a,c):
    optimalVector = priceVector(test, rho, a,c)
    profitWithGuessV = applyPriceVector(A, optimalVector, rho, a, c)
    trueProfit = applyPriceVector(A, priceVector(A, rho,a,c), rho, a, c)
    return trueProfit - profitWithGuessV
### AHHHHHHH


def getGaps(n, p, rho, a, c, i, results, n_trials):
    A, G = makeERGraph(n,p)
    print(n, p)
    results[i] = np.average( [getGap(A, makeSimilarGraph(G), rho, a, c) for j in range(n_trials)])
    print("Done,", i)
    return i


def getAverageGap(n, p, rho, a, c, i,  results, n_trials):
    A, G = makeERGraph(n,p)
    n = A.shape[0]
    trueProfit = applyPriceVector(A, priceVector(A, rho, a,c), rho, a, c)
    averageV = priceVector(makeSimilarGraph(G), rho, a,c)
    for i in range(n_trials-1):
        averageV += priceVector(makeSimilarGraph(G), rho, a, c)
    profit = applyPriceVector(A, (1/n_trials) * trueProfit, rho, a, c)
    return trueProfit-profit
