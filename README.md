# A Fair Comparison of Graph Neural Networks for Graph Classification

## Summary

TBD

## This repo

The library includes data and scripts to reproduce the experiments reported in the paper.

This research software is provided as is. If you happen to use or modify this code, please remember to cite the paper:

[*Federico Errica and Marco Podda, Davide Bacciu, Alessio Micheli: A Fair Comparison of Graph Neural Networks for Graph Classification. Proceedings of the 8th International Conference on Learning Representations.*](https://openreview.net/forum?id=HygDF6NFPB)

### Instructions

To reproduce the experiments, first preprocess datasets as follows:

`python PrepareDatasets.py DATA/CHEMICAL --dataset-name <name> --outer-k 10`
`python PrepareDatasets.py DATA/SOCIAL_1 --dataset-name <name> --use-one --outer-k 10`
`python PrepareDatasets.py DATA/SOCIAL_DEGREE --dataset-name <name> --use-degree --outer-k 10`

Where `<name>` is the name of the dataset. 

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
