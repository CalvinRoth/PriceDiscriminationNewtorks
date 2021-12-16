from helperFun import *
import tests
import multiprocessing as mp
import pandas as pd
from stopwatch import Stopwatch
from scipy import interpolate


if __name__ == '__main__':
        timer = Stopwatch();
        timer.start()
        # Paramters
        steps =14
        n = 1500
        n_delta = (n-400)/steps
        n_range = [ int(400+ i*n_delta) for i in range(0,steps)]
        p0 = 0.5/n
        p1 = 1.5* np.log(n)/n
        p = 1.3/n
        p_foo = lambda i : 1.3/n_range[i]
        delta = (p1-p0)/steps
        p_range = [p0+i*delta for i in range(steps)]
        rho = 0.9
        a = 4
        c = 2
        n_trials = 50
        n_trials_range = np.linspace(5,100, steps, dtype=int)
        outer_steps = 5
        """
        results = [ [0 for i in range(steps)] for j in range(outer_steps)]
        results1 = [0 for i in range(steps)]
        results2 = [0 for i in range(steps)]
        results3 = [0 for i in range(steps)]

        #results4 = [0 for i in range(steps)]
        # Test1 is apply the price vector of true graph to guessed graph and take average
        # Test2 is apply price vector of guess graph to true graph and take average
        # Test3 is collect all the price vectors of guesses and apply to true graph
        # All are reported with respect to the gap size := True Graph @ True Vector - Observed Profit
        #
        # I don't think the first test matters in this context. So I commented it out
        gapTest1 = getGaps
        gapTest2 = getGapsReverse
        gapTest3 = getAverageGap
        que = mp.Queue()
        results4 = mp.Array("f", range(steps))
        jobs = [];
        for i in range(steps):
                for j in range(outer_steps):
                        #currentParam4 = (n, p, rho, a, c, i, results[j], n_trials_range[i])
                        #getAverageGap(*currentParam4)
                        currentParam5 = (n, p_range[i], rho, a, c, i, results[j], n_trials)
                        fractionalGap(*currentParam5)
                #currentParams1 = (n, p_range[i], rho, a, c, i, results1, n_trials)
                #currentParams2 = (n_range[i], p_foo(i), rho, a, c, i, results2, n_trials)
                #currentParams3 = (n_range[i], p_foo(i), rho, a, c, i, results3, n_trials)
                #gapTest1(*currentParams1)
                #gapTest2(*currentParams2)
                # #gaptest3(*currentparams3)
                #job = mp.process(target=fractionalgap, args = (currentparam5))
                #job.start()
                #jobs.append(job)


        #for job in jobs:
        #        job.join()
        timer.stop()
        full_results = np.zeros((steps, outer_steps+1))
        full_results[:,0] = np.array(n_trials_range)
        for j in range(outer_steps):
                full_results[:,j+1] = np.array(results[j])

        plt.title("Profit Gap vs Trials Ran")
        #plt.plot(full_results[:,0], full_results[:,2], label="Apply guess vector", color="green")
        for j in range(outer_steps):
                plt.plot(full_results[:,0], full_results[:,j+1], label="Trial "+str(j+1))
        plt.xlabel("Trials Ran")
        plt.ylabel("Gap in profit")
        plt.legend()
        plt.show()
        #tabled = pd.DataFrame(full_results, columns = ["p value", "Test 1 ", "Test 2", "Test 3"])
        #tabled.to_csv("n_trials1_3_n.csv")
        """

        #resultsVar = varianceVector(n, p, rho, a, c, n_trials)
        #robustRes = robustNess(n, p, 0.05, rho,  a,c)
        #count_range = [i for i in range(n)]


        #results = tests.nTrialstest(n,p, n_trials_range, rho, a,c)
        #results1 = tests.sameparamProfitsVarTest(n,p_range, n_trials, rho, a, c)
        #results2 = tests.sameseqProfitsVarTest(n,p_range, n_trials, rho, a,c)
        #print(np.var(results1))
        #print(np.var(results2))
        #fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
        #axs[0].plot(p_range, results1)
        #axs[0].plot(p_range, results2)

        """results0 = tests.sameseqcomparePrices(n,1/n,n_trials, rho, a, c)
        results1 = tests.sameseqcomparePrices(n, 2/n, n_trials, rho, a,c)
        results2 = tests.sameseqcomparePrices(n, np.log(n)/n, n_trials, rho, a,c)
        var1 = np.var(results0, axis=1)
        var2 = np.var(results1, axis=1)
        var3 = np.var(results2, axis=1)

        # Filter out 0 var nodes for plotting
        var1 = [i for i in var1 if i!=0]
        var2 = [i for i in var2 if i!=0]
        var3 = [i for i in var3 if i!=0]
        hist1 = np.histogram(var1, bins=len(var1))
        hist2 = np.histogram(var2, bins=len(var2))
        hist3 = np.histogram(var3, bins=len(var3))

        plt.hist(var1, bins=n, label="1/n")
        plt.hist(var2, bins=n, label="2/n")
        plt.hist(var3, bins=n, label="log(n)/n")
        plt.legend()
        plt.title("Effect of p on node prices")"""


        n = 100
        n_trials = 30
        steps = 30
        counting = [i for i in range(n)]
        p = np.log(n)/n
        p0 = 1.1/n
        p1 = np.log(n)/n
        delta = (p1-p0)/steps
        p_space = [p0+(delta*i) for i in range(steps)]
        resSeq, resPar = tests.fractionalRegretOVerP(n,p0, p1, n_trials, steps, rho, a,c)
        plt.title("Fractional regret as a function of p")
        plt.xlabel("p")
        plt.ylabel("regret")
        plt.plot(p_space, resSeq, label="SameSeq")
        plt.plot(p_space, resPar, label="SameParam")
        plt.legend()
        plt.show()

        print(tests.sanityCheck(n,p,rho, a,c))

