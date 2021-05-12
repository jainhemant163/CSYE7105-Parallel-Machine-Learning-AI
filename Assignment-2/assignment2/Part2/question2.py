
import numpy as np
import pandas as pd
import multiprocessing as mp
import time
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt


df = pd.DataFrame(np.random.randint(3, 10, size=[20000,100]))
#print(df.head())

def funcs(row):
    return round(row[1]**2 + row[2]**2, 2)

with mp.Pool(1) as pool:
    starttime1 = time.time()
    result = pool.imap(funcs, df.itertuples(name=None), chunksize=10)
    output = [round(x, 2) for x in result]
    endtime1 = time.time() - starttime1

print(endtime1)
print(output)

with mp.Pool(2) as pool:
    starttime2 = time.time()
    result = pool.imap(funcs, df.itertuples(name=None), chunksize=10)
    output = [round(x, 2) for x in result]
    endtime2 = time.time() - starttime2

print(endtime2)
print(output)

with mp.Pool(4) as pool:
    starttime4 = time.time()
    result = pool.imap(funcs, df.itertuples(name=None), chunksize=10)
    output = [round(x, 2) for x in result]
    endtime4 = time.time() - starttime4

print(endtime4)
print(output)

with mp.Pool(8) as pool:
    starttime8 = time.time()
    result = pool.imap(funcs, df.itertuples(name=None), chunksize=10)
    output = [round(x, 2) for x in result]
    endtime8 = time.time() - starttime8

print(endtime8)
print(output)

processors = [1,2,4,8]
elapsed_Pool = [endtime1,endtime2,endtime4,endtime8]

plt.plot(processors,elapsed_Pool, label='Pool')
plt.xlabel('Processors')
plt.ylabel('Elapsed Time')
plt.legend()
plt.savefig('Elapsed_time.png',transparent=True, bbox_inches='tight')
plt.show()
