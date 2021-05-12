import multiprocessing as mp 

def common (a, b):

    list=set(a).intersection(b)
    return list

pool = mp.Pool(mp.cpu_count())
list_a = [[1, 2, 3], [5, 6, 7, 8], [10, 11, 12], [20, 21]]
list_b = [[2, 3, 4, 5], [6, 9, 10], [11, 12, 13, 14], [21, 24, 25]]

res = [pool.apply(common,args=(a, b)) for a, b in zip(list_a, list_b)] 
pool.close()
print("Common elements in both lists are: \n")
print (res[:10])
