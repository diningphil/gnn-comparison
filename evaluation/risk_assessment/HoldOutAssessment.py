import os
import json

from config.base import Config
from evaluation.dataset_getter import DatasetGetter
from log.Logger import Logger


class HoldOutAssessment:
    """
    Class implementing a sufficiently general framework to do model ASSESSMENT
    """

    def __init__(self, model_selector, exp_path, model_configs, max_processes=2):
        self.max_processes = max_processes
        self.model_configs = model_configs  # Dictionary with key:list of possible values
        self.model_selector = model_selector

        # Create the experiments folder straight away
        self.exp_path = exp_path
        self._HOLDOUT_FOLDER = os.path.join(exp_path, 'HOLDOUT_ASS')
        self._ASSESSMENT_FILENAME = 'assessment_results.json'

    def risk_assessment(self, experiment_class, debug=False, other=None):
        """
        :param experiment_class: the kind of experiment used
        :param debug:
        :return: An average over the outer test folds. RETURNS AN ESTIMATE, NOT A MODEL!!!
        """
        if not os.path.exists(self._HOLDOUT_FOLDER):
            os.makedirs(self._HOLDOUT_FOLDER)
        else:
            print("Folder already present! Shutting down to prevent loss of previous experiments")
            return

        self._risk_assessment_helper(experiment_class, self._HOLDOUT_FOLDER, debug, other)

    def _risk_assessment_helper(self, experiment_class, exp_path, debug=False, other=None):

        dataset_getter = DatasetGetter(None)

        best_config = self.model_selector.model_selection(dataset_getter, experiment_class, exp_path,
                                                          self.model_configs, debug, other)

        # Retrain with the best configuration and test
        experiment = experiment_class(best_config['config'], exp_path)

        # Set up a log file for this experiment (I am in a forked process)
        logger = Logger(str(os.path.join(experiment.exp_path, 'experiment.log')), mode='a')

        dataset_getter.set_inner_k(None)

        training_scores, test_scores = [], []

        # Mitigate bad random initializations
        for i in range(3):
            training_score, test_score = experiment.run_test(dataset_getter, logger, other)
            print(f'Final training run {i + 1}: {training_score}, {test_score}')

            training_scores.append(training_score)
            test_scores.append(test_score)

        training_score = sum(training_scores)/3
        test_score = sum(test_scores)/3

        logger.log('TR score: ' + str(training_score) + ' TS score: ' + str(test_score))

        with open(os.path.join(self._HOLDOUT_FOLDER, self._ASSESSMENT_FILENAME), 'w') as fp:
            json.dump({'best_config': best_config, 'HOLDOUT_TR': training_score, 'HOLDOUT_TS': test_score}, fp)
