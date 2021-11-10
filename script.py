from helperFun import *
import multiprocessing as mp
import time


tik = time.time()
# Paramters
n = 1200
p0 = 1/n
p1 = 2* np.log(n)/n
steps = 10
delta = (p1-p0)/steps
Ps = [p0+i*delta for i in range(steps)]
rho = 0.9
a = 4
c = 2
n_trials = 20


i = 0 
#pool = mp.Pool(Process=4)
#result = pool.apply_async(getGaps, (n, Ps[p], rho, a, c, p, average_gaps, n_trials))
#result.ready()
manager = mp.Queue()
average_gaps = manager.dict()

jobs = []
for p in range(steps):
	job = mp.Process(target=getGaps, args= (n, Ps[p], rho, a, c, p, average_gaps, n_trials))
	jobs.append(job)
#	job.start()
	#getGets(trueAdj, trueG, rho, a, c, p, average_gaps, n_trials)
	#print(i)
	#average_gaps.append(np.average([
	#	getGap(trueAdj, makeSimilarGraph(trueG), rho, a, c) 
 	#	for i in range(n_trials)
 	#	]))

#for job in jobs:
#	job.join()

tok = time.time()
print(tok-tik)

print(average_gaps)
plt.title("Gap with different p")
plt.plot(Ps, average_gaps)