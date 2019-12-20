from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from datasets import *
from models.graph_classifiers.DGCNN import DGCNN
from models.graph_classifiers.DeepMultisets import DeepMultisets
from models.graph_classifiers.MolecularFingerprint import MolecularFingerprint
from models.schedulers.ECCScheduler import ECCLR
from models.utils.EarlyStopper import Patience, GLStopper
from models.graph_classifiers.GIN import GIN
from models.graph_classifiers.DiffPool import DiffPool
from models.graph_classifiers.ECC import ECC
from models.graph_classifiers.GraphSAGE import GraphSAGE
from models.modules import (BinaryClassificationLoss, MulticlassClassificationLoss,
                            NN4GMulticlassClassificationLoss, DiffPoolMulticlassClassificationLoss)

from copy import deepcopy
from .utils import read_config_file


class ConfigError(Exception):
    pass


class Config:
    """
    Specifies the configuration for a single model.
    """
    datasets = {
        'NCI1': NCI1,
        'IMDB-BINARY': IMDBBinary,
        'IMDB-MULTI': IMDBMulti,
        'COLLAB': Collab,
        'REDDIT-BINARY': RedditBinary,
        'REDDIT-MULTI-5K': Reddit5K,
        'PROTEINS': Proteins,
        'ENZYMES': Enzymes,
        'DD': DD,
    }

    models = {
        'GIN': GIN,
        'ECC': ECC,
        "DiffPool": DiffPool,
        "DGCNN": DGCNN,
        "MolecularFingerprint": MolecularFingerprint,
        "DeepMultisets": DeepMultisets,
        "GraphSAGE": GraphSAGE
    }

    losses = {
        'BinaryClassificationLoss': BinaryClassificationLoss,
        'MulticlassClassificationLoss': MulticlassClassificationLoss,
        'NN4GMulticlassClassificationLoss': NN4GMulticlassClassificationLoss,
        'DiffPoolMulticlassClassificationLoss': DiffPoolMulticlassClassificationLoss,

    }

    optimizers = {
        'Adam': Adam,
        'SGD': SGD
    }

    schedulers = {
        'StepLR': StepLR,
        'ECCLR': ECCLR,
        'ReduceLROnPlateau': ReduceLROnPlateau
    }

    early_stoppers = {
        'GLStopper': GLStopper,
        'Patience': Patience
    }

    def __init__(self, **attrs):

        # print(attrs)
        self.config = dict(attrs)

        for attrname, value in attrs.items():
            if attrname in ['dataset', 'model', 'loss', 'optimizer', 'scheduler', 'early_stopper']:
                if attrname == 'dataset':
                    setattr(self, 'dataset_name', value)
                if attrname == 'model':
                    setattr(self, 'model_name', value)
                fn = getattr(self, f'parse_{attrname}')
                setattr(self, attrname, fn(value))
            else:
                setattr(self, attrname, value)

    def __getitem__(self, name):
        # print("attr", name)
        return getattr(self, name)

    def __contains__(self, attrname):
        return attrname in self.__dict__

    def __repr__(self):
        name = self.__class__.__name__
        return f'<{name}: {str(self.__dict__)}>'

    @property
    def exp_name(self):
        return f'{self.model_name}_{self.dataset_name}'

    @property
    def config_dict(self):
        return self.config

    @staticmethod
    def parse_dataset(dataset_s):
        assert dataset_s in Config.datasets, f'Could not find {dataset_s} in dictionary!'
        return Config.datasets[dataset_s]

    @staticmethod
    def parse_model(model_s):
        assert model_s in Config.models, f'Could not find {model_s} in dictionary!'
        return Config.models[model_s]

    @staticmethod
    def parse_loss(loss_s):
        assert loss_s in Config.losses, f'Could not find {loss_s} in dictionary!'
        return Config.losses[loss_s]

    @staticmethod
    def parse_optimizer(optim_s):
        assert optim_s in Config.optimizers, f'Could not find {optim_s} in dictionary!'
        return Config.optimizers[optim_s]

    @staticmethod
    def parse_scheduler(sched_dict):
        if sched_dict is None:
            return None

        sched_s = sched_dict['class']
        args = sched_dict['args']

        assert sched_s in Config.schedulers, f'Could not find {sched_s} in schedulers dictionary'

        return lambda opt: Config.schedulers[sched_s](opt, **args)

    @staticmethod
    def parse_early_stopper(stopper_dict):
        if stopper_dict is None:
            return None

        stopper_s = stopper_dict['class']
        args = stopper_dict['args']

        assert stopper_s in Config.early_stoppers, f'Could not find {stopper_s} in early stoppers dictionary'

        return lambda: Config.early_stoppers[stopper_s](**args)

    @staticmethod
    def parse_gradient_clipping(clip_dict):
        if clip_dict is None:
            return None
        args = clip_dict['args']
        clipping = None if not args['use'] else args['value']
        return clipping

    @classmethod
    def from_dict(cls, dict_obj):
        return Config(**dict_obj)


class Grid:
    """
    Specifies the configuration for multiple models.
    """

    def __init__(self, path_or_dict, dataset_name):
        self.configs_dict = read_config_file(path_or_dict)
        self.configs_dict['dataset'] = [dataset_name]
        self.num_configs = 0  # must be computed by _create_grid
        self._configs = self._create_grid()

    def __getitem__(self, index):
        return self._configs[index]

    def __len__(self):
        return self.num_configs

    def __iter__(self):
        assert self.num_configs > 0, 'No configurations available'
        return iter(self._configs)

    def _grid_generator(self, cfgs_dict):
        keys = cfgs_dict.keys()
        result = {}

        if cfgs_dict == {}:
            yield {}
        else:
            configs_copy = deepcopy(cfgs_dict)  # create a copy to remove keys

            # get the "first" key
            param = list(keys)[0]
            del configs_copy[param]

            first_key_values = cfgs_dict[param]
            for value in first_key_values:
                result[param] = value

                for nested_config in self._grid_generator(configs_copy):
                    result.update(nested_config)
                    yield deepcopy(result)

    def _create_grid(self):
        '''
        Takes a dictionary of key:list pairs and computes all possible permutations.
        :param configs_dict:
        :return: A dictionary generator
        '''
        config_list = [cfg for cfg in self._grid_generator(self.configs_dict)]
        self.num_configs = len(config_list)
        return config_list
