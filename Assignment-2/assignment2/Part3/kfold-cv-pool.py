import numpy as np
from multiprocessing import Pool
from sklearn import svm
from time import time


test = np.loadtxt("optdigits.txt", delimiter = ",")
X = test[:, 0:64]
y = test[:, 64]


def cvkfold(X, y, tuning_params, partitions, k):
    n_tuning_params = tuning_params.shape[0]

    partition = partitions[k]
    TRAIN = np.delete(np.arange(0, X.shape[0]), partition)
    TEST = partition
    X_train = X[TRAIN, :]
    y_train = y[TRAIN]

    X_test = X[TEST, :]
    y_test = y[TEST]

    accuracies = np.zeros(n_tuning_params)
    for i in range(0, n_tuning_params):
        svc = svm.SVC(C = tuning_params[i], kernel = "linear")
        accuracies[i] = svc.fit(X_train, y_train).score(X_test, y_test)
    return accuracies


K = 5
tuning_params = np.logspace(-6, -1, 10)
partitions = np.array_split(np.random.permutation([i for i in range(0, X.shape[0])]), K)

t1 = time()

num = 8
pool = Pool(num)
args = [(X, y, tuning_params, partitions, k) for k in range(0, K)]
Accuracies = np.array(pool.starmap(cvkfold, args))
pool.close()

print('\nPool Parallelism runs %0.3f seconds on %s on CPUs.\n' % ((time()-t1), num))

CV_accuracy = np.mean(Accuracies, axis = 0)
best_tuning_param = tuning_params[np.argmax(CV_accuracy)]

print('Best tuning param %0.6f.\n'% best_tuning_param)

