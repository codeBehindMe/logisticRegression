import numpy as np
import pandas as pd

from numutils import Nonlinearity


class BinaryLogisticBGD:
    """
    Binary classifier using logistic regression and batch gradient descent optimisation.
    """

    @staticmethod
    def _process_input(input):
        # type: (pd.DataFrame) -> np.ndarray

        """
        This method processes the input from pandas dataframe into numpy arrays.
        :param input: Pandas dataframe or numpy array of the inputs.
        :return: Numpy array
        """

        if type(input) is np.ndarray:
            return input

        return input.as_matrix().reshape(input.shape[0], input.shape[1])

    def initialize_weights(self, weightMatrix=None):
        # type: () -> None
        """
        Initializes the weight vector to use.
        :return: None
        """

        # Get the number of columns in the training features.
        # Initialise it column vector style.
        # [[0],
        #  [0],
        #  [0]]

        self.weights = np.zeros((self.tr_feat.shape[1], 1))
        self.bias = 0

        return

    @staticmethod
    def _activate_sigmoid_neuron(w, b, X):
        """
        Computes sigmoid activation.
        :param w: Weights
        :param b: Bias
        :param X: Batch of features.
        :return:
        """
        # type: (np.ndarray,float,np.ndarray,np.ndarray) -> np.ndarray
        return Nonlinearity.sigmoid(np.dot(w.T, X.T) + b)

    @staticmethod
    def _compute_cost(A, Y):
        """
        Computes cost based on the activation function and the class labels.
        :param A: Neuron activation for the batch.
        :param Y: Batch labels.
        :return: Cost.
        """

        _cost = (-1 / A.shape[1]) * np.sum((Y * np.log(A) + (1 - Y) * np.log(1 - A)))

        return

    def __init__(self, tr_feat, tr_lab, ts_feat, ts_lab):
        """
        Constructor
        :param tr_feat: Pandas dataframe or numpy array containing the training features.
        :param tr_lab:  Pandas dataframe or numpy array containing the training labels.
        :param ts_feat: Pandas dataframe or numpy array containing the test features.
        :param ts_lab:  Pandas dataframe or numpy array containing the test labels.
        """
        self.ts_lab = ts_lab
        self.ts_feat = ts_feat
        self.tr_lab = tr_lab
        self.tr_feat = tr_feat

    def train(self):
        pass

    def predict(self):
        pass


if __name__ == '__main__':
    x_train = np.random.randn(10, 3)

    weights = np.zeros((3, 1))

    y_train = Nonlinearity.sigmoid(np.sum(x_train * [-0.5, 1.24, 3.7], axis=1))
    y_train = (y_train.reshape(10, 1) > 0.5).astype(int)

    _classifier = BinaryLogisticBGD(x_train, y_train, None, None)

    _classifier.initialize_weights()

    _activation = _classifier._activate_sigmoid_neuron(_classifier.weights, _classifier.bias, x_train)

    _cost = _classifier._compute_cost(_activation, y_train)

    pass
