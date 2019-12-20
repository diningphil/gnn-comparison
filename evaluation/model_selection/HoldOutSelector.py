import os
import json
import concurrent.futures

from log.Logger import Logger


class HoldOutSelector:
    """
    Class implementing a sufficiently general framework to do model selection
    """

    def __init__(self, max_processes):
        self.max_processes = max_processes

        # Create the experiments folder straight away
        self._CONFIG_BASE = 'config_'
        self._CONFIG_FILENAME = 'config_results.json'
        self.WINNER_CONFIG_FILENAME = 'winner_config.json'

    def process_results(self, HOLDOUT_MS_FOLDER, no_configurations):

        best_vl = 0.

        for i in range(1, no_configurations+1):
            try:
                config_filename = os.path.join(HOLDOUT_MS_FOLDER, self._CONFIG_BASE + str(i),
                                               self._CONFIG_FILENAME)

                with open(config_filename, 'r') as fp:
                    config_dict = json.load(fp)

                vl = config_dict['VL_score']

                if best_vl <= vl:
                    best_i = i
                    best_vl = vl
                    best_config = config_dict

            except Exception as e:
                print(e)

        print('Model selection winner for experiment', HOLDOUT_MS_FOLDER, 'is config ', best_i, ':')
        for k in best_config.keys():
            print('\t', k, ':', best_config[k])

        return best_config

    def model_selection(self, dataset_getter, experiment_class, exp_path, model_configs, debug=False, other=None):
        """
        :param experiment_class: the kind of experiment used
        :param debug:
        :return: the best performing configuration on average over the k folds. TL;DR RETURNS A MODEL, NOT AN ESTIMATE!
        """
        HOLDOUT_MS_FOLDER = os.path.join(exp_path, 'HOLDOUT_MS')

        if not os.path.exists(HOLDOUT_MS_FOLDER):
            os.makedirs(HOLDOUT_MS_FOLDER)

        config_id = 0

        pool = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_processes)

        for config in model_configs:  # generate_grid(model_configs):

            # Create a separate folder for each experiment
            exp_config_name = os.path.join(HOLDOUT_MS_FOLDER, self._CONFIG_BASE + str(config_id + 1))
            if not os.path.exists(exp_config_name):
                os.makedirs(exp_config_name)

            json_config = os.path.join(exp_config_name, self._CONFIG_FILENAME)
            if not os.path.exists(json_config):
                if not debug:
                    pool.submit(self._model_selection_helper, dataset_getter, experiment_class, config,
                                exp_config_name, other)
                else:  # DEBUG
                    self._model_selection_helper(dataset_getter, experiment_class, config, exp_config_name,
                                                other)
            else:
                # Do not recompute experiments for this fold.
                print(f"Config {json_config} already present! Shutting down to prevent loss of previous experiments")
                continue

            config_id += 1

        pool.shutdown()  # wait the batch of configs to terminate

        best_config = self.process_results(HOLDOUT_MS_FOLDER, config_id)

        with open(os.path.join(HOLDOUT_MS_FOLDER, self.WINNER_CONFIG_FILENAME), 'w') as fp:
            json.dump(best_config, fp)

        return best_config

    def _model_selection_helper(self, dataset_getter, experiment_class, config, exp_config_name,
                                other=None):
        """
        :param dataset_getter:
        :param experiment_class:
        :param config:
        :param exp_config_name:
        :param other:
        :return:
        """

        # Create the experiment object which will be responsible for running a specific experiment
        experiment = experiment_class(config, exp_config_name)

        # Set up a log file for this experiment (run in a separate process)
        logger = Logger(str(os.path.join(experiment.exp_path, 'experiment.log')), mode='a')
        logger.log('Configuration: ' + str(experiment.model_config))

        config_filename = os.path.join(experiment.exp_path, self._CONFIG_FILENAME)

        # ------------- PREPARE DICTIONARY TO STORE RESULTS -------------- #

        selection_dict = {
            'config': experiment.model_config.config_dict,
            'TR_score': 0.,
            'VL_score': 0.,
        }

        dataset_getter.set_inner_k(None)  # need to stay this way

        training_score, validation_score = experiment.run_valid(dataset_getter, logger, other)

        selection_dict['TR_score'] = float(training_score)
        selection_dict['VL_score'] = float(validation_score)

        logger.log('TR Accuracy: ' + str(training_score) + ' VL Accuracy: ' + str(validation_score))

        with open(config_filename, 'w') as fp:
            json.dump(selection_dict, fp)
