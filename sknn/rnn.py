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

import keras

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM, GRU
from keras.optimizers import SGD


import sknn
import nn




class RecurrentLayer(nn.Layer):
    def __init__(
            self,
            type,
            outer_type = None,
            inner_type = None,
            warning=None,
            name=None,
            units=None,
            pieces=None,
            inner_pieces = None,
            weight_decay=None,
            dropout=None):

        assert warning is None,\
            "Specify layer parameters as keyword arguments, not positional arguments."

        if type not in ["GRU", "LSTM"]:
            raise NotImplementedError("Recurrent Layer type `%s` is not implemented." % type)

        if inner_type not in ['Rectifier', 'Sigmoid', 'Tanh']:
            raise NotImplementedError("Layer type `%s` is not implemented." % type)


        self.name = name
        self.type = type
        self.inner_type = inner_type
        self.outer_type = outer_type
        self.units = units
        self.pieces = pieces
        self.inner_pieces = inner_pieces
        self.weight_decay = weight_decay
        self.dropout = dropout

    def set_params(self, **params):
        """Setter for internal variables that's compatible with ``scikit-learn``.
        """
        for k, v in params.items():
            if k not in self.__dict__:
                raise ValueError("Invalid parameter `%s` for layer `%s`." % (k, self.name))
            self.__dict__[k] = v

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        copy = self.__dict__.copy()
        del copy['type']
        params = ", ".join(["%s=%r" % (k, v) for k, v in copy.items() if v is not None])
        return "<sknn.rnn.%s %s`%s`: %s>" % (self.__class__.__name__, self.type, self.inner_type, params)


class RecurrentNeuralNetwork(nn.NeuralNetwork):


    def _check_layer(self, layer, required, optional=[]):
        return True

    def _create_layer(self, name, layer, input_units):


        if layer.type == 'Rectifier':
            self._check_layer(layer, ['units'])
            return [Dense(input_dim=input_units,output_dim= layer.units, activation="relu")]


        if layer.type == 'Linear':
            self._check_layer(layer, ['units'])
            return [Dense(input_dim=input_units,output_dim= layer.units, activation="linear")]

        if layer.type == 'GRU':
            self._check_layer(layer, ['units'])
            conversion = {}
            conversion["Rectifier"] = "relu"

            return [GRU(input_dim=input_units,output_dim= layer.units, activation=conversion[layer.outer_type], inner_activation = conversion[layer.inner_type])]


    def _setup(self):
        self.initialised = False

    def is_initialized(self):
        return self.initialised

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


        if(not self.is_initialized()):
            print "Initialising"
            n_input = len(X_list[0][0])
            n_output = len(y_list[0])
            model = Sequential()
            #model.add(Activation('linear'))
            for i in range(len(self.layers)):

                if( i + 1 == len(self.layers)):
                    self.layers[i].units = n_output

                if(i == 0 ):
                    input_units = n_input
                else:
                    input_units = self.layers[i-1].units
                seq = self._create_layer("noname",self.layers[i],input_units)
                for s in seq:
                    model.add(s)
                self.initialised = True

            sgd = SGD(nesterov=True,momentum=0.9, decay=0.001,lr =0.01 )
            model.compile(loss='mse', optimizer=sgd)


        model.fit(X_list, y_list, batch_size=self.batch_size, nb_epoch=int(self.n_iter),verbose=True, show_accuracy=True)
        self.model = model









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


        return self.model.evaluate(X_list,y_list)



def generate_data(l = 11,max_pad = 20):
    g = lambda _: [([np.random.randint(1,2)*1.0]   ) for i in range(np.random.randint(4,l))]
    #print l
    p = lambda seq: seq + [[0]] * (max_pad - len(seq))
    X = [p(g(i)) for i in range(4000)]
    #print X
    y = []
    #print X
    for i in range(0, len(X)):
        x = X[i]

        c = x[0][0] - x[1][0]
        y_i = [c, 1]
        #print y_i
        y.append(y_i)
    X = np.array(X)
    y = np.array(y)
    print X.shape
    print y.shape
    # # exit()
    #print X
    return X, y



if __name__=="__main__":



    X,y = generate_data(5)

    layers = layers=[RecurrentLayer("GRU", outer_type= "Rectifier", inner_type = "Rectifier",  units=20),
                                    nn.Layer("Linear")]
    clf = RecurrentNeuralNetwork(layers=layers, n_iter=2)

    clf.fit(X,y)


    X,y = generate_data(7)
    score = clf.predict(X,y)
    print score, "7"

    X,y = generate_data(5)
    score = clf.predict(X,y)
    print score, "5"

    X,y = generate_data(100,100)
    score = clf.predict(X,y)
    print score, "3"

    # print y[:10]
    # print y_hat[:10]
































