from helperFun import *
import multiprocessing as mp
import time

if __name__ == '__main__':
        tik = time.time()
        # Paramters
        steps =150
        n_range = [i*200 for i in range(1,steps+1)]
        n = 1000
        p0 = 1.3/n
        p1 = 2* np.log(n)/n
        p = 1.3/n
        delta = (p1-p0)/steps
        p_range = [p0+i*delta for i in range(steps)]
        rho = 0.9
        a = 4
        c = 2
        n_trials = 20

        multiprocess = False
        max_threads = 8

        if(multiprocess):
                manager = mp.Queue(maxsize=max_threads)
                results = mp.Array("f", range(steps))
                jobs = []
        else:
                results = [0 for i in range(steps)]

        currentTest = getGaps

        print("N vertices", "probability of links")
        for i in range(steps):
                currentParams = (n_range[i], p, rho, a, c, i, results, n_trials)
                if(multiprocess):
                        job = mp.Process(target=currentTest, args=currentParams)
                        jobs.append(job)
                        job.start()
                else:
                        currentTest(*currentParams)

        if(multiprocess):
                for job in jobs:
                        job.join()

        tok = time.time()
        print(tok-tik)

        print(list(results))
        #plt.show()
        print(type(steps), type(results))
        plt.plot(n_range, results)
        plt.show()
