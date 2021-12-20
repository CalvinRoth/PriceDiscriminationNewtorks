```python
from __future__ import annotations
import numpy as np
import numpy.linalg as lin
import networkx as nx
import scipy
import scipy.sparse.linalg as slin
import matplotlib.pyplot as plt
```

#  Basic Functionality 

Here we supply the essential functions needed for experiments. This includes the ability to find the term relating to the spectral norm which is abstracted in a function because it may change. 

Also included a generators for ER graphs, the ability to compute the optimal price vector for a graph and the optimal profit of a graph if we were to apply this price vector. 


```python
# Linear algebra
def specNorm(A: np.matrix) -> float:
    return lin.norm(A, ord=np.inf)
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


def centralty(A: np.matrix, rho: float, alpha,k) -> np.matrix:
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
    valid = 1
    if (np.min(consumption) < 0):
        valid = 0
    return ((v - (c * ones)).T @ consumption)[0, 0], valid


def priceVector(A: np.matrix, rho: float, a: int | float, c: int | float,k = None) -> np.matrix:
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
    central = centralty(A, rho, alpha,k)  # This should be A not A + A.T because of how centralty function is designed
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


```

# Metrics
Here are functions used "one level up" in terms of abstraction from finding the price vectors. This includes things like computing the fractional regret of a applying a given vector v to a graph instad of its true optimal price vector. 


```python

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

```

# Tests
Here are the collection of various tests we can run. 


```python
def profitDistribution(n : int, p : float,  n_trials : int, rho : float, a : int | float ,c : int | float):
    results = np.zeros(n_trials)
    A_true, G_true = makeERGraph(n, p)
    trueP = optimalProfit(A_true,n, a, c, rho)
    res_par = np.zeros((n_trials,))
    res_seq = np.zeros((n_trials,))
    valid_par = 0
    valid_seq = 0    
    for j in range(n_trials):
        A_par, G_par = makeERGraph(n,p)
        A_seq = makeSimilarGraph(G_true)
        v_par = priceVector(A_par, rho, a, c)
        v_seq = priceVector(A_seq, rho, a, c)
        res_par[j],b0 = applyPriceVector(A_true, v_par, rho, a, c)
        res_seq[j],b1 = applyPriceVector(A_true, v_seq, rho, a, c)
        valid_par += b0
        valid_seq += b1
        
    return [res_par, res_seq], [valid_par/n_trials, valid_seq/n_trials]


def regretDistribution(n : int, p : float,  n_trials : int, rho : float, a : int | float ,c : int | float):
    results = np.zeros(n_trials)
    A_true, G_true = makeERGraph(n, p)
    trueP = optimalProfit(A_true,n, a, c, rho)
    res_par = np.zeros((n_trials,))
    res_seq = np.zeros((n_trials,))
    valid_par = 0
    valid_seq = 0    
    for j in range(n_trials):
        A_par, G_par = makeERGraph(n,p)
        A_seq = makeSimilarGraph(G_true)
        v_par = priceVector(A_par, rho, a, c)
        v_seq = priceVector(A_seq, rho, a, c)
        res_par[j],b0 = fractionalRegret(A_true, v_par, n, rho, a, c)
        res_seq[j],b1 = fractionalRegret(A_true, v_seq, n, rho, a, c)
        valid_par += b0
        valid_seq += b1
        
    return [res_par, res_seq], [valid_par/n_trials, valid_seq/n_trials]

def walkDistribution(n : int, p : float,  n_trials : int, rho : float, a : int | float ,c : int | float,k : int):
    results = np.zeros(n_trials)
    res = np.zeros((n_trials,))
    for j in range(n_trials):
        A_true, G_true = makeERGraph(n, p)
        trueP = optimalProfit(A_true,n, a, c, rho)
        v = priceVector(A_true, rho, a, c,k)
        res[j] = fractionalRegret(A_true, v, n, rho, a, c)[0]
    return res

```

# Effects of different Norms
Reporting for n=1500, rho=0.9, p=sqrt(log(n))/n, a=5, c=4 the variance of profits and the fraction of valid samples. 

## Infinity Norm


```python
def specNorm(A: np.matrix) -> float:
    return lin.norm(A, ord=np.inf)
n = 1500
p = np.sqrt(np.log(n))/n
n_trials = 50
rho = 0.9
a = 5
c = 4
[res_par, res_seq], [ratio_par, ratio_seq] = profitDistribution(n,p,n_trials,  rho, a, c)
[regret_par, regret_seq] = regretDistribution(n,p,n_trials,  rho, a, c)
print("Infinity Norm")
print("Same Parameter var", np.var(res_par)), print("Same Sequence var", np.var(res_seq))
print("Same Parameter mean", np.mean(res_par)), print("Same Sequence mean", np.mean(res_seq))
print("Same Parameter ratio", ratio_par), print("Same Sequence ratio", ratio_seq)
```

    Infinity Norm
    Same Parameter var 7.933976166720137
    Same Sequence var 0.0002606741040087665
    Same Parameter mean 265.15448143188024
    Same Sequence mean 284.4032453002981
    Same Parameter ratio 0.96
    Same Sequence ratio 1.0





    (None, None)



## Frobenieus Norm
It appears this norm is 2 large so the network effect is small


```python
def specNorm(A: np.matrix) -> float:
    return lin.norm(A)
n = 1500
p = np.sqrt(np.log(n))/n
n_trials = 50
rho = 0.9
a = 5
c = 4
[res_par, res_seq], [ratio_par, ratio_seq] = profitDistribution(n,p,n_trials,  rho, a, c)
[regret_par, regret_seq] = regretDistribution(n,p,n_trials,  rho, a, c)
print("Infinity Norm")
print("Same Parameter var", np.var(res_par)), print("Same Sequence var", np.var(res_seq))
print("Same Parameter mean", np.mean(res_par)), print("Same Sequence mean", np.mean(res_seq))
print("Same Parameter ratio", ratio_par), print("Same Sequence ratio", ratio_seq)
```

    Infinity Norm
    Same Parameter var 5.693779946153654e-05
    Same Sequence var 3.157721105772635e-09
    Same Parameter mean 198.09179454896
    Same Sequence mean 198.31363155686822
    Same Parameter ratio 1.0
    Same Sequence ratio 1.0





    (None, None)



## 1 Norm


```python
def specNorm(A: np.matrix) -> float:
    return lin.norm(A, ord=1)
n = 1500
p = np.sqrt(np.log(n))/n
n_trials = 50
rho = 0.9
a = 5
c = 4
[res_par, res_seq], [ratio_par, ratio_seq] = profitDistribution(n,p,n_trials,  rho, a, c)
[regret_par, regret_seq] = regretDistribution(n,p,n_trials,  rho, a, c)
print("One Norm")
print("Same Parameter var", np.var(res_par)), print("Same Sequence var", np.var(res_seq))
print("Same Parameter mean", np.mean(res_par)), print("Same Sequence mean", np.mean(res_seq))
print("Same Parameter ratio", ratio_par), print("Same Sequence ratio", ratio_seq)
```

    One Norm
    Same Parameter var 12.830250183054028
    Same Sequence var 0.00010939270473722128
    Same Parameter mean 257.6831732487298
    Same Sequence mean 275.913781629323
    Same Parameter ratio 1.0
    Same Sequence ratio 1.0





    (None, None)



## Nuclear norm


```python
def specNorm(A: np.matrix) -> float:
    return lin.norm(A, ord="nuc")
n = 1500
p = np.sqrt(np.log(n))/n
n_trials = 50
rho = 0.9
a = 5
c = 4
#[res_par, res_seq], [ratio_par, ratio_seq] = profitDistribution(n,p,n_trials,  rho, a, c)
#[regret_par, regret_seq] = regretDistribution(n,p,n_trials,  rho, a, c)
#print("One Norm")
#print("Same Parameter var", np.var(res_par)), print("Same Sequence var", np.var(res_seq))
#print("Same Parameter mean", np.mean(res_par)), print("Same Sequence mean", np.mean(res_seq))
#print("Same Parameter ratio", ratio_par), print("Same Sequence ratio", ratio_seq)
```

## 2 Norm
This is very confusing and worrying. The results show that under the 2 norm same parameter graphs are likely to be malformed often. Also this code takes much longer to run than the others


```python
def specNorm(A: np.matrix) -> float:
    return lin.norm(A, ord=2)
n = 1500
p = np.sqrt(np.log(n))/n
n_trials = 50
rho = 0.9
a = 5
c = 4
"""
[res_par, res_seq], [ratio_par, ratio_seq] = profitDistribution(n,p,n_trials,  rho, a, c)
[regret_par, regret_seq] = regretDistribution(n,p,n_trials,  rho, a, c)
print("2 Norm")
print("Same Parameter var", np.var(res_par)), print("Same Sequence var", np.var(res_seq))
print("Same Parameter mean", np.mean(res_par)), print("Same Sequence mean", np.mean(res_seq))
print("Same Parameter ratio", ratio_par), print("Same Sequence ratio", ratio_seq)"""
```




    '\n[res_par, res_seq], [ratio_par, ratio_seq] = profitDistribution(n,p,n_trials,  rho, a, c)\n[regret_par, regret_seq] = regretDistribution(n,p,n_trials,  rho, a, c)\nprint("2 Norm")\nprint("Same Parameter var", np.var(res_par)), print("Same Sequence var", np.var(res_seq))\nprint("Same Parameter mean", np.mean(res_par)), print("Same Sequence mean", np.mean(res_seq))\nprint("Same Parameter ratio", ratio_par), print("Same Sequence ratio", ratio_seq)'



We conclude that seemingly any choice of norm besides 2-norm is sufficient. It is unclear to me if I need to weight these norms in someway to be bigger/smamer.

# Limited Walk Size
One question we have asked is if knowing the degrees of each matrix is sufficient perhaps knowing the reach of each node in two steps is better because it specifies the graph more. Here instead of generating new graphs specificied by this information we will look at the regret using only the true graph but with truncated centralities


```python
def specNorm(A: np.matrix) -> float:
    return lin.norm(A, ord=np.inf)
def centralty(A: np.matrix, rho: float, alpha, k=None) -> np.matrix:
    """

    Parameters
    ----------
    A : np matrix
    rho : network effect
    k : size of walk to take in network

    Returns
    -------
    Centrality vector as described in paper
    """
    n = A.shape[0]

    if k is None:

        ident = np.eye(n, n)
        ones = np.ones((n, 1))
        ApA = A + A.T
        eig = specNorm(ApA)
        alpha = rho / eig
        central = lin.inv(ident - (alpha * ApA))
        central = central @ ones  # Checked.  this > 0
        return central
    else:
        one = np.ones((n,1))
        ApA = A + A.T
        alpha = rho / specNorm(ApA)
        total = alpha * ApA
        base = alpha * np.eye(n, n)
        for i in range(1,k+1):
            total = total + np.linalg.matrix_power(base, i)
        return total @ one   
```


```python
n = 1500
p = np.sqrt(np.log(n))/n
n_trials = 50
rho = 0.9
a = 5
c = 4
k = 1
v = walkDistribution(n, p, n_trials, rho, a, c,k)
print("k is ", k)
print("Mean Loss", np.mean(v))
print("Variance", np.var(v))
k = 2 
v = walkDistribution(n, p, n_trials, rho, a, c,k)
print("k is ", k)
print("Mean Loss", np.mean(v))
print("Variance", np.var(v))
k = None
v = walkDistribution(n, p, n_trials, rho, a, c,k)
print("k is ", k)
print("Mean Loss", np.mean(v))  
print("Variance", np.var(v))
```

    k is  1
    Mean Loss 0.017556982847047202
    Variance 6.635231756986075e-06
    k is  2
    Mean Loss 0.017339977211125745
    Variance 6.951887580322704e-06
    k is  None
    Mean Loss 4.4408920985006264e-17
    Variance 9.762153702110021e-32


As we see there is very little advanage in increasing the step size and as a ratio a sizable tighening of variance 


