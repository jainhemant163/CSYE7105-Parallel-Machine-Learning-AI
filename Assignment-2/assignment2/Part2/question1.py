from numpy.lib import financial
import pandas as pd
import numpy as np
from multiprocessing import Pool
import time
import matplotlib.pyplot as plt

np.random.seed(1234)

df = pd.DataFrame(np.random.randint(3, 10, size=[20000, 100]))


def sum_squares(x):
    new_list = [y*y for y in x]
    return sum(new_list)


speedup = []
cpus = [1, 2, 4, 8]


def plot_speedup_vs_cpus(cpus_nums):
    for i in cpus_nums:
        start = time.perf_counter()
        pool = Pool(processes=i)
        result = pool.map(sum_squares, [df[df.columns[column]]
                                        for column in df.columns])
        pool.close()

        speedup.append(time.perf_counter()-start)
    plt.plot(cpus, speedup)
    plt.title('Speedup Vs # CPUs used')
    plt.xlabel('CPUs Used')
    plt.ylabel('Speedup (sec)')
    plt.show(block=True)

if __name__ == '__main__':
    plot_speedup_vs_cpus(cpus)
