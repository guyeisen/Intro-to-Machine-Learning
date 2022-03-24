#################################
# Your name: Guy Eisenberg
#################################
import timeit

import numpy as np
import matplotlib.pyplot as plt
import intervals


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """
    I_P = [(0, 0.2), (0.4, 0.6), (0.8, 1)]

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        X = np.sort(np.random.uniform(low=0, high=1, size=m))
        positive_probs = [0.2, 0.8]
        negative_probs = [0.9, 0.1]
        possible_values = [0, 1]
        Y = [np.random.choice(possible_values, p=positive_probs) if self.is_in_I_P(x)
             else np.random.choice(possible_values, p=negative_probs) for x in X]
        return np.column_stack((X,Y))

    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        empiricalErr = np.zeros(int(m_last/step)-1)
        trueErr = np.zeros(int(m_last/step)-1)
        for t in range(T):
            for i in range(1,int(m_last/step)):
                XY = self.sample_from_D(i*step)
                best_intervals, empirical_error = intervals.find_best_interval(XY[:,0], XY[:,1], k)
                true_error = self.get_true_error(best_intervals)

                empiricalErr[i-1] += empirical_error/(i*step)
                trueErr[i-1] += true_error
        empiricalErr = empiricalErr/T
        trueErr = trueErr/T

        self.plot_my_graph(title="Empirical VS True error as function of n",
                           X=list(range(int(m_last/step)-1)),
                           Y1=empiricalErr,
                           Y2=trueErr,
                           xlabel="n",
                           ylabel="error",
                           graph1label="Empirical Error",
                           graph2label="True Error")


    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        XY = self.sample_from_D(m)
        K = list(range(k_first,k_last,step))

        empErr = np.zeros(len(K))
        trueErr = np.zeros(len(K))
        for k in K:
            best_intervals, empirical_error = intervals.find_best_interval(XY[:, 0], XY[:, 1], k)
            true_error = self.get_true_error(best_intervals)

            empErr[k-1] = empirical_error/m
            trueErr[k-1] = true_error

        self.plot_my_graph(title="Empirical vs True error as function of k",
                           X=K,
                           Y1=empErr,
                           Y2=trueErr,
                           xlabel="k",
                           ylabel="error",
                           graph1label="Empirical Error",
                           graph2label="True Error")

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Run the experiment in (c).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        XY = self.sample_from_D(m)
        K = list(range(k_first, k_last, step))

        empErr = np.zeros(len(K))
        trueErr = np.zeros(len(K))
        penalty = np.zeros(len(K))
        sum = np.zeros(len(K))
        for k in K:
            best_intervals, empirical_error = intervals.find_best_interval(XY[:, 0], XY[:, 1], k)
            true_error = self.get_true_error(best_intervals)

            empErr[k - 1] = empirical_error / m
            trueErr[k - 1] = true_error
            penalty[k-1] = self.get_penalty(k,m)
            sum[k-1] = empErr[k - 1] + penalty[k-1]

        self.plot_my_graph(title="Empirical vs True error as function of k",
                           X=K,
                           Y1=empErr,
                           Y2=trueErr,
                           Y3=penalty,
                           Y4=sum,
                           xlabel="k",
                           ylabel="error",
                           graph1label="Empirical Error",
                           graph2label="True Error",
                           graph3label="Penalty",
                           graph4label="Sum of empirical+penalty")


    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """

        XY = self.sample_from_D(m)
        np.random.shuffle(XY)
        test_sample = XY[:int(m/5)]
        train_sample = np.array(sorted(XY[int(m/5):], key=lambda x:x[0]))


        K = list(range(1, 10, 1))
        empErr = np.zeros(len(K))
        exacttrueErr = np.zeros(len(K))
        testErr = np.zeros(len(K))
        for k in K:
            best_intervals, empirical_error = intervals.find_best_interval(train_sample[:, 0], train_sample[:, 1], k)
            exact_true_error = self.get_true_error(best_intervals)
            empErr[k - 1] = empirical_error / (m/5)
            exacttrueErr[k - 1] = exact_true_error
            testErr[k-1] = self.count_errors(best_intervals, test_sample,int(m/5)) /(m/5)

        self.plot_my_graph(title="Cross validation",
                           X=K,
                           Y1=empErr,
                           Y2=exacttrueErr,
                           Y3=testErr,
                           xlabel="k",
                           ylabel="error",
                           graph1label="Empirical Error",
                           graph2label="True Error",
                           graph3label="Test Error")

    #################################
    # Place for additional methods

    def count_errors(self, hipo_intervals, test_samples, length):
        sum = 0
        for i in range(length):
            if self.is_in_range(test_samples[i,0], hipo_intervals) and test_samples[i,1] == 1:
                sum+=0
            elif self.is_in_range(test_samples[i,0], hipo_intervals) and test_samples[i,1] == 0:
                sum+=1
            elif not self.is_in_range(test_samples[i,0], hipo_intervals) and test_samples[i,1] == 0:
                sum+=0
            elif not self.is_in_range(test_samples[i,0], hipo_intervals) and test_samples[i,1] == 1:
                sum+=1
        return sum

    def is_in_I_P(self, x):
        return self.is_in_range(x, self.I_P)

    def is_in_range(self, x, I):
        for l, u in I:
            if l <= x <= u:
                return True
        return False

    def get_all_intervals(self, I):
        I_mixed = []
        for l, u in self.I_P + I:
            I_mixed.append(l)
            I_mixed.append(u)
        I_mixed.sort()
        all_intervals = []
        for i in range(len(I_mixed) - 1):
            l = I_mixed[i]
            u = I_mixed[i + 1]
            if (self.is_in_I_P((l + u) / 2) and self.is_in_range((l + u) / 2, I)):  # answer the same
                all_intervals.append((l, u, 1, 1))
            elif (not self.is_in_I_P((l + u) / 2) and not self.is_in_range((l + u) / 2, I)):
                all_intervals.append((l, u, 1, 0))
            elif (not self.is_in_I_P((l + u) / 2) and self.is_in_range((l + u) / 2, I)):
                all_intervals.append((l, u, 0, 1))
            elif (self.is_in_I_P((l + u) / 2) and not self.is_in_range((l + u) / 2, I)):
                all_intervals.append((l, u, 0, 0))

        return all_intervals

    def get_true_error(self, I):
        true_error = 0
        all_intervals = self.get_all_intervals(I)
        for l, u, equal, answer in all_intervals:
            if equal:
                true_error += (u - l) * (0.2 if answer == 1 else 0.1)
            else:
                true_error += (u - l) * (0.9 if answer == 0 else 0.1)
        return true_error

    def get_penalty(self, k, n):
        return np.sqrt((2*k + np.log(20))/n)


    def plot_my_graph(self, title, X, Y1, Y2, ylabel, xlabel, graph1label, graph2label, Y3=None, graph3label=None, Y4=None, graph4label=None):
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.plot(X, Y1, label=graph1label)
        plt.plot(X, Y2, label=graph2label)
        if Y3 is not None:
            plt.plot(X, Y3, label=graph3label)
        if Y4 is not None:
            plt.plot(X, Y4, label=graph4label)
        plt.legend()
        plt.show()
    #################################


if __name__ == '__main__':
    ass = Assignment2()
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500)
