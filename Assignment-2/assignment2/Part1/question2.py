import os  
import numpy as np                                                                     
import multiprocessing as mp

process = ('script1.py', 'script2.py', 'script3.py')                      

def parallel(process):                                                             
    os.system('python {}'.format(process))                                      

pool = mp.Pool(processes=3)                                                        
pool.map(parallel, process) 
