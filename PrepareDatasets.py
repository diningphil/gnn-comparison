import argparse

from datasets import *


DATASETS = {
    'REDDIT-BINARY': RedditBinary,
    'REDDIT-MULTI-5K': Reddit5K,
    'COLLAB': Collab,
    'IMDB-BINARY': IMDBBinary,
    'IMDB-MULTI': IMDBMulti,
    'NCI1': NCI1,
    'ENZYMES': Enzymes,
    'PROTEINS': Proteins,
    'DD': DD
}


def get_args_dict():
    parser = argparse.ArgumentParser()

    parser.add_argument('DATA_DIR',
                        help='where to save the datasets')
    parser.add_argument('--dataset-name', dest='dataset_name',
                        choices=DATASETS.keys(), default='all', help='dataset name [Default: \'all\']')
    parser.add_argument('--outer-k', dest='outer_k', type=int,
                        default=10, help='evaluation folds [Default: 10]')
    parser.add_argument('--inner-k', dest='inner_k', type=int,
                        default=None, help='model selection folds [Default: None]')
    parser.add_argument('--use-one', action='store_true',
                        default=False, help='use 1 as feature')
    parser.add_argument('--use-degree', dest='use_node_degree', action='store_true',
                        default=False, help='use degree as feature')
    parser.add_argument('--no-kron', dest='precompute_kron_indices', action='store_false',
                        default=True, help='don\'t precompute kron reductions')

    return vars(parser.parse_args())


def preprocess_dataset(name, args_dict):
    dataset_class = DATASETS[name]
    if name == 'ENZYMES':
        args_dict.update(use_node_attrs=True)
    dataset_class(**args_dict)


if __name__ == "__main__":
    args_dict = get_args_dict()

    print(args_dict)

    dataset_name = args_dict.pop('dataset_name')
    if dataset_name == 'all':
        for name in DATASETS:
            preprocess_dataset(name, args_dict)
    else:
        preprocess_dataset(dataset_name, args_dict)