import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from multiprocessing import Pool, Process
from sklearn import svm
import time


test = np.loadtxt("optdigits.txt", delimiter = ",")
X = test[:, 0:64]
y = test[:, 64]

# Plot some of the digits
#fig = plt.figure(figsize=(8, 6))
#fig.tight_layout()
#for i in range(0, 20):
#    ax = fig.add_subplot(5, 5, i + 1)
#    ax.imshow(X[i].reshape((8,8)), cmap = "Greys", vmin = 0, vmax = 16)
#plt.show()


def cvkfold(X, y, tuning_params, partitions, k):
    n_tuning_params = tuning_params.shape[0]

    partition = partitions[k]
    Train = np.delete(np.arange(0, X.shape[0]), partition)
    Test = partition
    X_train = X[Train, :]
    y_train = y[Train]

    X_test = X[Test, :]
    y_test = y[Test]

    accuracies = np.zeros(n_tuning_params)
    for i in range(0, n_tuning_params):
        svc = svm.SVC(C = tuning_params[i], kernel = "linear")
        accuracies[i] = svc.fit(X_train, y_train).score(X_test, y_test)
    return accuracies


K = 5
tuning_params = np.logspace(-6, -1, 10)
partitions = np.array_split(np.random.permutation([i for i in range(0, X.shape[0])]), K)

t1 = time.time()

for k in range(0, K):
    Accuracies = cvkfold(X, y, tuning_params, partitions, k)
ts1 = time.time() - t1
print('Serial runs %0.3f seconds.' %ts1)

CV_accuracy = np.mean(Accuracies, axis = 0)
best_tuning_param = tuning_params[np.argmax(CV_accuracy)]
print('Best tuning param %0.6f.'% best_tuning_param)

#using Pool

#using 2 processors
pool = Pool(processes=2) 
t1 =time.time()
args = [(X, y, tuning_params, partitions, k) for k in range(0, K)]
Accuracies = np.array(pool.starmap(cvkfold, args))
tp2 = time.time()-t1
print("Pool for 2 processors runs %0.3f seconds. "%tp2)
pool.close()
CV_accuracy = np.mean(Accuracies, axis = 0)
best_tuning_param = tuning_params[np.argmax(CV_accuracy)]
print("Best tuning parmeter is %0.6f." %best_tuning_param)

#for 4 processors
pool = Pool(processes=4) 
t1 =time.time()
args = [(X, y, tuning_params, partitions, k) for k in range(0, K)]
Accuracies = np.array(pool.starmap(cvkfold, args))
tp4 = time.time()-t1
print("Pool for 4 processors runs %0.3f seconds. "%tp4)
pool.close()
CV_accuracy = np.mean(Accuracies, axis = 0)
best_tuning_param = tuning_params[np.argmax(CV_accuracy)]
print("Best tuning parmeter is %0.6f." %best_tuning_param)

#for 8 processors
pool = Pool(processes = 8) 
t1 =time.time()
args = [(X, y, tuning_params, partitions, k) for k in range(0, K)]
Accuracies = np.array(pool.starmap(cvkfold, args))
tp8 = time.time()-t1
print("Pool for 8 processors runs %0.3f seconds. "%tp8)
pool.close()
CV_accuracy = np.mean(Accuracies, axis = 0)
best_tuning_param = tuning_params[np.argmax(CV_accuracy)]
print("Best tuning parmeter is %0.6f." %best_tuning_param)

                                                      

#using process

#using 2 processors
p = Pool(processes = 2)
t1=time.time()
args = [(X, y, tuning_params, partitions, k) for k in range(0, K)]
Accuracies= np.array(p.Process(target=cvkfold, args=args))

tpr2 = time.time()-t1
print("Process class for 2 processors runs %0.3f seconds."%tpr2)


CV_accuracy = Accuracies
best_tuning_param = tuning_params[np.argmax(CV_accuracy)]
print("Best tuning parmeter is %0.6f." %best_tuning_param)

#using 4 processors
p = Pool(processes = 4)
t1=time.time()
args = [(X, y, tuning_params, partitions, k) for k in range(0, K)]
Accuracies= np.array(p.Process(target=cvkfold, args=args))

tpr4 = time.time()-t1
print("Process class for 4 processors runs %0.3f seconds."%tpr4)


CV_accuracy = Accuracies
best_tuning_param = tuning_params[np.argmax(CV_accuracy)]
print("Best tuning parmeter is %0.6f." %best_tuning_param)

#using 8 processors
p = Pool(processes = 8)
t1=time.time()
args = [(X, y, tuning_params, partitions, k) for k in range(0, K)]
Accuracies= np.array(p.Process(target=cvkfold, args=args))

tpr8 = time.time()-t1
print("Process class for 8 processors runs %0.3f seconds."%tpr8)


CV_accuracy = Accuracies
best_tuning_param = tuning_params[np.argmax(CV_accuracy)]
print("Best tuning parmeter is %0.6f." %best_tuning_param)

processors = [1,2,4,8]
elapsed_ser = [ts1,ts1,ts1,ts1]
elapsed_Pool = [0,tp2,tp4,tp8]
elapsed_Process= [0,tpr2,tpr4,tpr8]

plt.plot(processors,elapsed_ser, label='Serial')
plt.plot(processors,elapsed_Pool, label='Pool')
plt.plot(processors,elapsed_Process, label='Process')
plt.xlabel('Processors')
plt.ylabel('Elapsed Time')
plt.legend()
plt.savefig('Elapsed_time.png',transparent=True, bbox_inches='tight')
plt.show()
