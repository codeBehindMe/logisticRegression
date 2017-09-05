from unittest import TestCase
from linearmodel import BinaryLogisticBGD
import numpy as np
import pandas as pd


class TestBinaryLogisticBGD(TestCase):
    def test_initialize_weights(self):

        """
        Test if the input processor method correctly processes the numpy array input.
        :return:
        """

        tr_set = np.random.randn(10, 3)  # Make a sample training set.

        _output = BinaryLogisticBGD._process_input(tr_set)

        if _output.shape[1] != tr_set.shape[1]:
            self.fail()

    def test_initialise_weights_with_pandas(self):

        """
        Test if the input processor correctly processes the pandas dataframe inputs.
        :return:
        """

        tr_set = pd.DataFrame(np.random.randn(10, 3))

        _output = BinaryLogisticBGD._process_input(tr_set)

        if _output.shape[1] != tr_set.shape[1]:
            self.fail()
