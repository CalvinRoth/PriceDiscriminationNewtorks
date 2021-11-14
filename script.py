from helperFun import *
import multiprocessing as mp
import pandas as pd
from stopwatch import Stopwatch

if __name__ == '__main__':
        timer = Stopwatch();
        timer.start()
        # Paramters
        steps =75
        n = 1000
        n_delta = (n-400)/steps
        n_range = [ int(400+ i*n_delta) for i in range(1,steps+1)]
        p0 = 1.3/n
        p1 = 2* np.log(n)/n
        p = 1.3/n
        p_foo = lambda i : 1.3/n_range[i]
        delta = (p1-p0)/steps
        p_range = [p0+i*delta for i in range(steps)]
        rho = 0.9
        a = 4
        c = 2
        n_trials = 20

        results1 = [0 for i in range(steps)]
        results2 = [0 for i in range(steps)]
        results3 = [0 for i in range(steps)]
        # Test1 is apply the price vector of true graph to guessed graph and take average
        # Test2 is apply price vector of guess graph to true graph and take average
        # Test3 is collect all the price vectors of guesses and apply to true graph
        # All are reported with respect to the gap size := True Graph @ True Vector - Observed Profit
        #
        # I don't think the first test matters in this context. So I commented it out
        gapTest1 = getGaps
        gapTest2 = getGapsReverse
        gapTest3 = getAverageGap
        for i in range(steps):
                #currentParams1 = (n, p_range[i], rho, a, c, i, results1, n_trials)
                currentParams2 = (n, p_range[i], rho, a, c, i, results2, n_trials)
                currentParams3 = (n, p_range[i], rho, a, c, i, results3, n_trials)
                #gapTest1(*currentParams1)
                gapTest2(*currentParams2)
                gapTest3(*currentParams3)
                timer.lap(i)


        timer.stop()
        full_results = np.zeros((steps, 4))
        print(np.array(p_range).shape)
        full_results[:,0] = np.array(p_range)
        full_results[:,1] = np.array(results1)
        full_results[:,2] = np.array(results2)
        full_results[:,3] = np.array(results3)

        plt.title("Profit Gap vs N")
        plt.plot(full_results[:,0], full_results[:,2], label="Apply guess vector", color="green")
        plt.plot(full_results[:,0], full_results[:,3], label="Apply average guess vector", color="blue")
        plt.xlabel("p")
        plt.ylabel("Gap in profit")
        plt.legend()
        plt.show()
        tabled = pd.DataFrame(full_results, columns = ["p value", "Test 1 ", "Test 2", "Test 3"])
        tabled.to_csv("n100p1_3_n.csv")
