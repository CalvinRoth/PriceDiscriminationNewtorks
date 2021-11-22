from helperFun import *
import multiprocessing as mp
import pandas as pd
from stopwatch import Stopwatch

if __name__ == '__main__':
        timer = Stopwatch();
        timer.start()
        # Paramters
        steps =10
        n = 500
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
        n_trials = 20
        n_trials_range = np.linspace(5,100, steps, dtype=int)

        results = [ [0 for i in range(steps)] for j in range(5)]
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
                for j in range(5):
                        currentParam5 = (n, p_range[i], rho, a, c, i, results[j], n_trials)
                        fractionalGap(*currentParam5)
                #currentParams1 = (n, p_range[i], rho, a, c, i, results1, n_trials)
                #currentParams2 = (n_range[i], p_foo(i), rho, a, c, i, results2, n_trials)
                #currentParams3 = (n_range[i], p_foo(i), rho, a, c, i, results3, n_trials)
                #currentParam4 = (n , p, rho , a, c, i, results4, n_trials_range[i])
                #gapTest1(*currentParams1)
                #gapTest2(*currentParams2)
                # #gaptest3(*currentparams3)
                #job = mp.process(target=fractionalgap, args = (currentparam5))
                #job.start()
                #jobs.append(job)


        #for job in jobs:
        #        job.join()

        timer.stop()
        full_results = np.zeros((steps, 6))
        full_results[:,0] = np.array(p_range)
        for j in range(5):
                full_results[:,j+1] = np.array(results[j])

        plt.title("Profit Gap vs N")
        #plt.plot(full_results[:,0], full_results[:,2], label="Apply guess vector", color="green")
        for j in range(5):
                plt.plot(full_results[:,0], full_results[:,j+1], label="Trial "+str(j+1))
        plt.xlabel("P-Value")
        plt.ylabel("Gap in profit")
        plt.legend()
        plt.show()
        #tabled = pd.DataFrame(full_results, columns = ["p value", "Test 1 ", "Test 2", "Test 3"])
        #tabled.to_csv("n_trials1_3_n.csv")
