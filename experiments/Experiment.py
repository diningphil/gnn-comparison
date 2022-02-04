#
# Copyright (C)  2020  University of Pisa
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
import random
from config.base import Config


class Experiment:
    """
    Experiment provides a layer of abstraction to avoid that all models implement the same interface
    """

    def __init__(self, model_configuration, exp_path):
        self.model_config = Config.from_dict(model_configuration)
        self.exp_path = exp_path

    def run_valid(self, get_train_val, logger, other=None):
        """
        This function returns the training and validation accuracy. DO WHATEVER YOU WANT WITH VL SET,
        BECAUSE YOU WILL MAKE PERFORMANCE ASSESSMENT ON A TEST SET
        :return: (training accuracy, validation accuracy)
        """
        raise NotImplementedError('You must implement this function!')

    def run_test(self, get_train_val, get_test, logger, other=None):
        """
        This function returns the training and test accuracy
        :return: (training accuracy, test accuracy)
        """
        raise NotImplementedError('You must implement this function!')


class ToyExperiment(Experiment):

    def __init__(self, model_configuration, exp_path):
        super(ToyExperiment, self).__init__(model_configuration, exp_path)

    def run_valid(self, get_train_val, logger, other=None):
        """
        This function returns the training and validation or test accuracy
        :return: (training accuracy, validation/test accuracy)
        """
        return random.uniform(0, 100), random.uniform(0, 100)

    def run_test(self, get_train_val, logger, get_test, other=None):
        """
        This function returns the training and test accuracy. DO NOT USE THE TEST FOR ANY REASON
        :return: (training accuracy, test accuracy)
        """
        return random.uniform(0, 100), random.uniform(0, 100)
