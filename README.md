# A Fair Comparison of Graph Neural Networks for Graph Classification (ICLR 2020)

## Summary

The library includes data and scripts to reproduce the experiments reported in the paper.

This research software is provided as is. If you happen to use or modify this code, please remember to cite the paper:

[Federico Errica, Marco Podda, Davide Bacciu, Alessio Micheli: *A Fair Comparison of Graph Neural Networks for Graph Classification*](https://openreview.net/pdf?id=HygDF6NFPB). *Proceedings of the 8th International Conference on Learning Representations (ICLR 2020).*

If you are interested in an introduction to Deep Graph Networks, check this out:

[Davide Bacciu, Federico Errica, Alessio Micheli, Marco Podda: *A Gentle Introduction to Deep Learning for Graphs*](https://arxiv.org/abs/1912.12693). *Under Review*

### Instructions

To reproduce the experiments, first preprocess datasets as follows:

`python PrepareDatasets.py DATA/CHEMICAL --dataset-name <name> --outer-k 10`

`python PrepareDatasets.py DATA/SOCIAL_1 --dataset-name <name> --use-one --outer-k 10`

`python PrepareDatasets.py DATA/SOCIAL_DEGREE --dataset-name <name> --use-degree --outer-k 10`

Where `<name>` is the name of the dataset. Then, substitute the split (json) files with the ones provided in the `data_splits` folder.

Please note that dataset folders should be organized as follows:

    CHEMICAL:
        NCI1
        DD
        ENZYMES
        PROTEINS
    SOCIAL[_1 | _DEGREE]:
        IMDB-BINARY
        IMDB-MULTI
        REDDIT-BINARY
        REDDIT-MULTI-5K
        COLLAB

Then, you can launch experiments by typing:

`cp -r DATA/[CHEMICAL|SOCIAL_1|SOCIAL_DEGREE]/<name> DATA`
`python Launch_Experiments.py --config-file <config> --dataset-name <name> --result-folder <your-result-folder> --debug`

Where `<config>` is your config file (e.g. config_BaselineChemical.yml), and `<name>` is the dataset name chosen as before.

### Troubleshooting

The installation of Pytorch Geometric depends on other libraries (torch_scatter, torch_cluster, torch_sparse) that have to be installed separately and before torch_geometric. Do not use pip install -r requirements.txt because it will not work. Please refer to the [official instructions](https://github.com/rusty1s/pytorch_geometric) to install the required libraries.

If you would like PyTorch not to spawn multiple threads for each process (**highly recommended**), append ``export OMP_NUM_THREADS=1`` to your .bashrc file.

