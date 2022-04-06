#################################
# Your name: Guy Eisenberg
#################################


import numpy as np
import numpy.random
from matplotlib import pyplot as plt
from numpy import reshape
from sklearn.datasets import fetch_openml
import sklearn.preprocessing

"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


HINGE = "hinge"
LOG = "log"



def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements SGD for hinge loss.
    """
    n = len(data[0])
    w_t = np.zeros(n)
    eta_t = eta_0
    for t in range(1, T):
        eta_t = eta_0 / t
        i = np.random.randint(low=0, high=n)
        if loss_func(HINGE, w_t, labels[i], data[i]) > 0:
            w_t = SGD_step(HINGE, eta_t, w_t, C, labels[i], data[i])
        else:
            w_t = SGD_step(HINGE,eta_t, w_t)
    return w_t

def SGD_log(data, labels, eta_0, T, plot=False):
    """
    Implements SGD for log loss.
    """
    n = len(data[0])
    w_t = np.zeros(n)
    eta_t = eta_0
    norms = np.zeros(T-1)
    for t in range(1, T):
        norms[t-1] = np.linalg.norm(w_t)
        eta_t = eta_0 / t
        i = np.random.randint(low=0, high=n)
        if loss_func(LOG, w_t, labels[i], data[i]) > 0:
            w_t = SGD_step(LOG, eta_t, w_t, 1, labels[i], data[i])
        else:
            w_t = SGD_step(LOG, eta_t, w_t)
    if plot:
        plot_my_graph(title="Norm of classifier as function of iteration",
                  X=range(1,T),
                  Y1=norms,
                  ylabel="Norm",
                  xlabel="Iteration")
    return w_t


#################################

# Place for additional code

def sign(w,x):
    if np.dot(w,x) >= 0:
        return 1
    return -1

def gradient(method, x, y, w):
    if method == HINGE:
        return -y*x
    elif method == LOG:
        exp = np.exp(-y*np.dot(x,w))
        return -y*(exp/(1+exp))*x

def SGD_step(method, eta, w, C=0, y=0, x=0):
    return (1 - eta) * w - eta * C * gradient(method, x, y, w)


def loss_func(method, w_t, y, x):
    if method == HINGE:
        return 1 - y * np.dot(w_t, x)
    elif method == LOG:
        exp = np.exp(-y * np.dot(x, w_t))
        return 1 - np.log(1+exp)



######### section a ###########

repetition = 10
train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()

def calc_averaged_accuracy_vector(i, validation_accuracy_vector ,w , average_over=10):
    for t in range(average_over):
        validation_accuracy_vector[i] += sum(
            1 for j, x in enumerate(validation_data) if sign(w, x) == validation_labels[j]) / len(validation_data)
    validation_accuracy_vector[i] = validation_accuracy_vector[i] / average_over


def get_best_eta(method):
    etas = np.logspace(-5, 3, num=1000)
    validation_accuracy = np.zeros(len(etas))
    for i, eta in enumerate(etas):
        if method == HINGE:
            w = SGD_hinge(train_data, train_labels, C=1, eta_0=eta, T=1000)
        elif method == LOG:
            w = SGD_log(train_data, train_labels, eta_0=eta, T=1000)
        calc_averaged_accuracy_vector(i, validation_accuracy, w)
    plot_my_graph(title="Accuracy as function of Eta",
                  X=etas,
                  Y1=validation_accuracy,
                  ylabel="Accuracy",
                  xlabel="Eta")
    return etas[np.argmax(validation_accuracy)]

def get_best_C_and_eta(method):
    eta = get_best_eta(method)
    print(eta)
    C_vec = np.logspace(-5,4,num=1000)
    validation_accuracy = np.zeros(len(C_vec))
    for i, C in enumerate(C_vec):
        if method == HINGE:
            w = SGD_hinge(train_data, train_labels, C=C, eta_0=eta, T=1000)
        elif method == LOG:
            w = SGD_log(train_data, train_labels, eta_0=eta, T=1000)
        calc_averaged_accuracy_vector(i, validation_accuracy, w)
    plot_my_graph(title="Accuracy as function of C with best eta",
                 X=C_vec,
                 Y1=validation_accuracy,
                 ylabel="Accuracy",
                 xlabel="C")
    return eta, C_vec[np.argmax(validation_accuracy)]

def get_w(method):
    if method == HINGE:
        eta, C = get_best_C_and_eta(method)
        w = SGD_hinge(train_data, train_labels, C=C, eta_0=eta, T=20000)
    else :
        eta = get_best_eta(method)
        w = SGD_log(train_data, train_labels, eta_0=eta, T=20000)

    plt.imshow(reshape(w,(28,28)), interpolation='nearest')
    plt.show()
    accuracy = sum(1 for i, x in enumerate(test_data) if sign(w, x) == test_labels[i])/len(test_data)
    print(f'Accuracy is {accuracy*100}%')

def plot_my_graph(title, X, Y1, ylabel, xlabel):
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.plot(X, Y1, 'o')
    plt.legend()
    plt.show()


#################################
get_w(HINGE)
get_w(LOG)
