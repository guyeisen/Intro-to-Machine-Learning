import numpy
import numpy as np
import matplotlib.pyplot as plt

def get_empiriacal_mean_vector_binomial(N=200000, n=20, p=0.5):
    """
    returns empirical mean vector from Bernulli(0.5)
    """
    mean_vec = np.zeros(N)
    for i in range(N-1):
        sanmples_vector = np.random.binomial(1, p, n)
        mean_vec[i] = np.sum(sanmples_vector) / n
    return mean_vec

def count_bigger_than(X, p, epsilon):
    count = 0
    for X_i in X:
        if np.abs(X_i - p) > epsilon:
            count +=1
    return count


def calulate_axises(p=0.5, N=200000, K=50):
    X = get_empiriacal_mean_vector_binomial(N=N, p=p)
    probabilities = np.zeros(K)
    epsilons = numpy.linspace(0,1, K)
    for i in range(len(epsilons)):
        probabilities[i] = count_bigger_than(X, p, epsilons[i]) / N
    return epsilons, probabilities

def get_hoeffding_axis(x):
    hoff = np.ze

x , y = calulate_axises()
hoeffding = 2*np.e**(-2*len(x)*x**2)
plt.title("Empirical probability vs Hoeffding bound")
plt.ylabel("probablity")
plt.xlabel("epsilon")
plt.plot(x ,y, label='Empirical')
plt.plot(x, hoeffding, label='Hoeffding')
plt.legend()
plt.show()

print()
