import numpy as np
import multiprocessing
import timeit

def start():
    a = np.random.rand(1000, 1000)
    b = np.random.rand(1000, 1000)
    np.multiply(a,b)
start2 = timeit.timeit()
for i in range(500):
    start()
end = timeit.timeit()
print(end-start2)
start2 = timeit.timeit()
pool = multiprocessing.Pool(multiprocessing.cpu_count())
liste = [pool.apply_async(start, ()) for i in range(500)]
[p.get() for p in liste]
end = timeit.timeit()
print(end - start2)
