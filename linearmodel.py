import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numutils import Nonlinearity


class BinaryLogisticBGD:
    """
    Binary classifier using logistic regression and batch gradient descent optimisation.
    """

    @staticmethod
    def _process_reshape_feature_tensor(features):
        # type: (pd.DataFrame) -> np.ndarray
        """
        This method processes the feature vectors and reshapes them into a matrix that has features rowise and the instances columnwise.
            x(1) x(2) ... x(i)
        x1
        x2
        .
        .
        .
        xn
        :param features: Features (Pandas dataframe or otherwise).
        :return: Numpy n-dimensional array containing the features for mat-mul.
        """
        # If no input given, return input.
        if features is None: return features

        if isinstance(features, np.ndarray):
            return features.reshape(features.shape[0], -1).T

        if isinstance(features, pd.DataFrame):
            return features.as_matrix().reshape(features.shape[0], -1).T

        raise TypeError()  # TODO: Implement a better error here ?????

    @staticmethod
    def _process_reshape_label_tensor(labels):
        # type: (pd.DataFrame) -> np.ndarray
        """
        Re-orient row-wise instances of labels to column-wise instances of labels.
        :param labels: Pandas data-frame or numpy array containing the labels.
        :return: Numpy array with labels.
        """

        # If no input given, return input.
        if labels is None: return labels

        if isinstance(labels, np.ndarray):
            return labels.reshape(labels.shape[0], -1).T
        if isinstance(labels, pd.DataFrame):
            return labels.as_matrix().reshape(labels.shape[0], -1).T

        raise TypeError()  # TODO: Implement a better error here ?????

    @staticmethod
    def _activate_sigmoid_neuron(w, b, X):
        """
        Computes sigmoid activation.
        :param w: Weights
        :param b: Bias
        :param X: Batch of features.
        :return:
        """
        # type: (np.ndarray,float,np.ndarray) -> np.ndarray
        return Nonlinearity.sigmoid(np.dot(w.T, X) + b)

    @staticmethod
    def _compute_cost(A, Y):
        """
        Computes cost based on the activation function and the class labels.
        :param A: Neuron activation for the batch.
        :param Y: Batch labels.
        :return: Cost.
        """

        return (-1 / A.shape[1]) * np.sum((Y * np.log(A) + (1 - Y) * np.log(1 - A)))

    @staticmethod
    def _compute_gradients(X, A, Y):
        # type: (np.ndarray,np.ndarray,np.ndarray) -> object
        """
        Computes gradients from the training labels, Activations and the labels.
        :param X: Training features.
        :param A: Activations.
        :param Y: Labels.
        :return: Weight Gradients, Bias gradient.
        """
        m = X.shape[1]
        a_y = (A - Y)

        dw = (1 / m) * np.dot(X, a_y.T)
        db = (1 / m) * np.sum(a_y)

        return dw, db

    def __init__(self, tr_feat, tr_lab, ts_feat, ts_lab):
        """
        Constructor
        -----------

        :param tr_feat: Pandas dataframe or numpy array containing the training features. The training features are expected with features column-wise and instances row-wise.
        :param tr_lab:  Pandas dataframe or numpy array containing the training labels.
        :param ts_feat: Pandas dataframe or numpy array containing the test features.
        :param ts_lab:  Pandas dataframe or numpy array containing the test labels.
        """

        # The re-orientation happens at instantiation since all mat-mul and other ops are referencing processed vectors / matrices in their local orientation.
        self.ts_lab = self._process_reshape_label_tensor(ts_lab)
        self.ts_feat = self._process_reshape_feature_tensor(ts_feat)
        self.tr_lab = self._process_reshape_label_tensor(tr_lab)
        self.tr_feat = self._process_reshape_feature_tensor(tr_feat)

        # Initialise a weight matrix with zeroes and a bias with zero.
        # The weight matrix is instantiated column-wise. [W1,W2,...,Wn]
        self.weights = np.zeros((self.tr_feat.shape[0], 1))
        self.bias = 0.0

        # Containers
        self.params = None  # Final model parameters.
        self.gradients = None  # Final model gradients.
        self.cost_iter = []  # Container for cost ~ iter information.

    def _initialize_weights(self, weightVector=None, bias=None):
        # type: (np.ndarray,float) -> None
        """
        Set a custom weight vector to initialize with.
        :param Weight Vector
        :param Bias
        :return: None
        """

        # If no params give, initialise to zero.
        self.weights = np.zeros((self.tr_feat.shape[0], 1))
        self.bias = 0.0

        # Check that the weight vector is the correct size for the training features.
        if weightVector.shape[1] != self.tr_feat.shape[0]:
            raise IndexError("Weight vector does not match the size of the features.")

        self.weights = weightVector
        self.bias = 0.0

        return

    def _batch_descent_optimise(self, max_epochs=1000, learning_rate=0.001):

        """
        Carries out batch gradient descent on the data.
        :param max_epochs: Maximum number of iterations.
        :param learning_rate: learning rate.
        :return: None
        """

        # Main loop
        # Iterate through to the max_epochs and update the weights.
        for i in range(max_epochs):
            # Feed forward
            activations = self._activate_sigmoid_neuron(self.weights, self.bias, self.tr_feat)
            cost = self._compute_cost(activations, self.tr_lab)
            self.cost_iter.append((i, cost))
            # Back propagate.
            dw, db = self._compute_gradients(self.tr_feat, activations, self.tr_lab)

            # Update weights.
            self.weights = self.weights - learning_rate * dw
            self.bias = self.bias - learning_rate * db

        self.params = np.append(self.weights, self.bias)

        return

    def train(self, max_epochs=1000, learning_rate=0.001,weightVector=None,bias=None):

        self._batch_descent_optimise(max_epochs, learning_rate)

    def predict(self):
        pass

    @property
    def get_cost_iter(self):
        return pd.DataFrame(self.cost_iter, columns=['iteration', 'cost'])


if __name__ == '__main__':
    x_train = np.random.randn(10, 3)

    y_train = Nonlinearity.sigmoid(np.sum(x_train * [-0.5, 1.24, 3.7], axis=1))
    y_train = (y_train.reshape(10, 1) > 0.5).astype(int)

    weights = np.zeros((3, 1))

    _classifier = BinaryLogisticBGD(x_train, y_train, None, None)

    _activation = _classifier._activate_sigmoid_neuron(_classifier.weights, _classifier.bias, _classifier.tr_feat)

    _cost = _classifier._compute_cost(_activation, y_train)

    dw, db = _classifier._compute_gradients(_classifier.tr_feat, _activation, _classifier.tr_lab)

    _classifier.train()

    cost_iter = _classifier.get_cost_iter
    plt.plot(cost_iter['iteration'], cost_iter['cost'])

    _classifier.train()

    plt.show()

    pass
