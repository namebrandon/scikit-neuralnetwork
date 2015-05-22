"""\
A simple belief network
=================================================
"""
print(__doc__)

__author__ = 'Spyros Samothrakis'

import sys
from sklearn.metrics import mean_squared_error as mse
import numpy as np


# The neural network uses the `sknn` logger to output its information.
import logging
logging.basicConfig(format="%(message)s", level=logging.WARNING, stream=sys.stdout)
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
from copy import deepcopy

import sknn.mlp as mlp

class RNNRegressor():
    """Regressor.
    """

    def __init__(self):
        pass








    def fit(self, X_list, y_list):
        """Fit the neural network to the given data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_inputs)
            Training vectors as real numbers, where n_samples is the number of
            samples and n_inputs is the number of input features.

        y : array-like, shape (n_samples, n_outputs)
            Target values as real numbers, either as regression targets or
            label probabilities for classification.

        Returns
        -------
        self : object
            Returns this instance.
        """
        # X is a sequence

        ## initialise

        pass






    def predict(self, X_list, y_list):
        """Calculate predictions for specified inputs.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_inputs)
            The input samples as real numbers.

        Returns
        -------
        y : array, shape (n_samples, n_outputs)
            The predicted values as real numbers.
        """


        pass


def generate_data(test = False):
    g = lambda _: [np.array([np.random.randint(-1,1)*1.0]) for i in range(5)]
    #print l
    X = [g(i) for i in range(1000)]
    #print X
    y = []
    #print X
    for i in range(0, len(X)):
        x = X[i]

        c = x[0] - x[1]
        y_i = [[None] for i in range(len(x)-1)] + [c]
        #print y_i
        y.append(y_i)

    return X, y



if __name__=="__main__":

    #
    layers = [mlp.Layer("Maxout", units=8, pieces=2),mlp.Layer("Maxout", units=8, pieces=2), mlp.Layer("Linear")]
    # mlp.Classifier(
    #     layers=[mlp.Layer("maxout", units=units), mlp.Layer(output)], random_state=1,
    #     n_iter=iterations, n_stable=iterations, regularize=regularize,
    #     dropout_rate=dropout, learning_rule=rule, learning_rate=alpha)



    X,y = generate_data()
    clf = RNNRegressor(layers=layers, hidden_state_variables=20, n_iter=100, learning_rate=0.001 , learning_rule = "momentum", batch_size=10)

    clf.fit(X,y)
    X_list, y_list = generate_data()
    X_mod, y_mod = clf.predict(X_list, y_list)

    y_2d_orig = np.array(np.array(y_list).transpose(2,0,1)[0].T[-1])
    y_2d_rec = np.array(y_mod).transpose(2,0,1)[0].T[-1]

    print "Unseen data MSE", mse(y_2d_orig,y_2d_rec)
    # print y[:10]
    # print y_hat[:10]
































