from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', as_frame=False)
data = mnist['data']
labels = mnist['target']

import numpy.random

idx = numpy.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]

import matplotlib.pyplot as plot
import numpy as np
import math


##########################  a  ##########################
def distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


def get_new_max(candidates, query_img):
    max = 0
    max_index = 0
    for i in range(len(candidates)):
        dist = distance(query_img, candidates[i][0])
        if dist >= max:
            max = dist
            max_index = i
    return max, max_index


def extract_labels_from_tuples(neighbors):
    lables = []
    for tup in neighbors:
        lables.append(int(tup[1]))
    return lables


def get_k_nearest_neighbors(train_images, labels, query_image, k):
    max_dist = math.inf
    max_dist_index = 0
    neighbors_candidate = []
    for i in range(len(train_images)):
        dist = distance(train_images[i], query_image)
        if dist < max_dist:
            if len(neighbors_candidate) == k:  # need to replace with the max one
                neighbors_candidate[max_dist_index] = (train_images[i], labels[i])
                max_dist, max_dist_index = get_new_max(neighbors_candidate, query_image)
            else:  # meanning len(neighbors_candidate) < k, can be pushed without taking somebody out
                neighbors_candidate.append((train_images[i], labels[i]))
        elif dist >= max_dist:
            if len(neighbors_candidate) == k:
                pass  # nothing to do
            else:  # len(neighbors_candidate) < k
                # add it to candidates
                neighbors_candidate.append(train_images[i])
                max_dist_index = len(neighbors_candidate) - 1
    return neighbors_candidate


def predict_query(train_images, labels, query_image, k):
    """
    using k-NN algorithm
    """
    neighbors_candidates = get_k_nearest_neighbors(train_images, labels, query_image, k)
    labels_of_candidates = extract_labels_from_tuples(neighbors_candidates)
    label_values, label_counts = np.unique(labels_of_candidates, return_counts=True)
    prediction_label = label_values[np.argmax(label_counts)]
    return prediction_label


##################### b #################################
def accuracy_of_prediction(train_images, train_labels, test_images, test_labels, k, n):
    good_predictions = 0
    train_images = train_images[:n]
    train_labels = train_labels[:n]

    for i in range(len(test_images)):
        predictiond_label = predict_query(train_images, train_labels, test_images[i], k)
        if predictiond_label == int(test_labels[i]):
            good_predictions += 1
    accuracy = good_predictions / len(test_images)
    return accuracy


accuracy_section_b = accuracy_of_prediction(train, train_labels, test, test_labels, 10, 1000)
print(f'accuracy of algorithm for the first 1000 train images: {accuracy_section_b * 100}%')


######################## c ####################################

def plot_section_c():
    n = 1000
    K = list(range(1, 101))
    Y = []

    for k in range(len(K)):
        y = accuracy_of_prediction(train, train_labels, test, test_labels, k + 1, n)
        Y.append(y)
    plot.xlabel("k")
    plot.ylabel("Accuracy")

    plot.plot(K, Y, color="darkorchid")
    plot.show()


plot_section_c()


############################ d ################################

def plot_section_d():
    Y = []
    N = list(range(100, 5001, 100))
    for n in N:
        y = accuracy_of_prediction(train, train_labels, test, test_labels, 1, n)
        Y.append(y)
        print(f'iteration {n} got accuracy of {y * 100}%')

    plot.xlabel("n")
    plot.ylabel("Accuracy")
    plot.plot(N, Y, color="darkorchid")
    plot.show()


plot_section_d()
