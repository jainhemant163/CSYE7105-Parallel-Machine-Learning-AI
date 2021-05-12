import multiprocessing as mp

def normalize(lc):
    minimum = min(lc)
    maximum = max(lc)
    ans = [(i - minimum)/(maximum-minimum) for i in lc]
    return ans

list_c = [[2, 3, 4, 5], [6, 9, 10, 12], [11, 12, 13, 14], [21, 24, 25, 26]]

pool = mp.Pool(mp.cpu_count())
results = [pool.apply(normalize, args=(l, )) for l in list_c]
pool.close()
print("Normalized values are \n")    
print(results[:10])
