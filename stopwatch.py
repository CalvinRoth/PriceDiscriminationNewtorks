import time
class Stopwatch():
    def __init__(self):
        self.start_time = 0
        self.last_recorded = 0

    def start(self):
        self.last_recorded = time.time()
        self.start_time = self.last_recorded

    def lap(self, i):
        old = self.last_recorded
        self.last_recorded = time.time()
        print("Current Iteration: ", i , "Time in current lap:", self.last_recorded-old, " Total time:", self.last_recorded - self.start_time)

    def stop(self):
        self.last_recorded = time.time()
        print(" Total time: ", self.last_recorded - self.start_time)
