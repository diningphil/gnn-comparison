import os
import torch

from config.base import Grid, Config

from evaluation.model_selection.HoldOutSelector import HoldOutSelector
from evaluation.risk_assessment.K_Fold_Assessment import KFoldAssessment
from experiments.EndToEndExperiment import EndToEndExperiment


def main(config_file, dataset_name,
         outer_k, outer_processes, inner_k, inner_processes, result_folder, debug=False):

    # Needed to avoid thread spawning, conflicts with multi-processing. You may set a number > 1 but take into account
    # the number of processes on the machine
    torch.set_num_threads(1)

    experiment_class = EndToEndExperiment

    model_configurations = Grid(config_file, dataset_name)
    model_configuration = Config(**model_configurations[0])

    exp_path = os.path.join(result_folder, f'{model_configuration.exp_name}_assessment')

    model_selector = HoldOutSelector(max_processes=inner_processes)
    risk_assesser = KFoldAssessment(outer_k, model_selector, exp_path, model_configurations,
                                    outer_processes=outer_processes)

    risk_assesser.risk_assessment(experiment_class, debug=debug)
