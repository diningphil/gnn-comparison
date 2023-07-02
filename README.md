# A Fair Comparison of Graph Neural Networks for Graph Classification (ICLR 2020)

## Summary

The library includes data and scripts to reproduce the experiments reported in the paper.

#### If you happen to use or modify this code, please remember to cite our paper:

[Federico Errica, Marco Podda, Davide Bacciu, Alessio Micheli: *A Fair Comparison of Graph Neural Networks for Graph Classification*](https://openreview.net/pdf?id=HygDF6NFPB). *Proceedings of the 8th International Conference on Learning Representations (ICLR 2020).*

    @inproceedings{errica_fair_2020,
	    title = {A fair comparison of graph neural networks for graph classification},
	    booktitle = {Proceedings of the 8th {International} {Conference} on {Learning} {Representations} ({ICLR})},
	    author = {Errica, Federico and Podda, Marco and Bacciu, Davide and Micheli, Alessio},
	    year = {2020}
    }
--

## Updated Table with Results (CHEMICAL)
|                             | D\&D           | NCI1            | PROTEINS        |
|-----------------------------|-------------------------|--------------------------|--------------------------|
| Baseline                    | $\mathbf{78.4}\pm 4.5 $ | $69.8 \pm 2.2 $          | $\mathbf{75.8} \pm 3.7 $ |
| DGCNN                       | $76.6 \pm 4.3 $         | $76.4 \pm 1.7 $          | $72.9 \pm 3.5 $          |
| DiffPool                    | $75.0 \pm 3.5 $         | $76.9 \pm 1.9 $          | $73.7 \pm 3.5 $          |
| ECC                         | $72.6 \pm 4.1 $         | $76.2 \pm 1.4 $          | $72.3 \pm 3.4 $          |
| GIN                         | $75.3 \pm 2.9 $         | $\mathbf{80.0} \pm 1.4 $ | $73.3 \pm 4.0 $          |
| GraphSAGE                   | $72.9 \pm 2.0 $         | $76.0 \pm 1.8 $          | $73.0 \pm 4.5 $          |
| [CGMM](https://www.jmlr.org/papers/volume21/19-470/19-470.pdf)              | $74.9 \pm 3.4 $         | $76.2 \pm 2.0$           | $74.0 \pm 3.9$           |
| [ECGMM](https://ieeexplore.ieee.org/document/9533430/)                      | $73.9 \pm4.1$           | $78.5 \pm 1.7$           | $73.3 \pm 4.1$           |
| [iCGMM<sub>*f*</sub>](https://proceedings.mlr.press/v162/castellana22a/castellana22a.pdf) | $75.1 \pm 3.8$          | $76.4 \pm1.4$            | $73.2 \pm 3.9$           |
| [GSPN](https://arxiv.org/pdf/2305.10544.pdf) | - | $76.6 \pm 1.9$ | - |
 
## Updated Table with Results (SOCIAL + degree)
|                             | IMDB-B         | IMDB-M          | REDDIT-B       | REDDIT-5K       | COLLAB         |
|-----------------------------|-------------------------|--------------------------|-------------------------|--------------------------|-------------------------|
| Baseline                   | $70.8 \pm 5.0 $         | $\mathbf{49.1} \pm 3.5 $ | $82.2 \pm 3.0 $         | $52.2 \pm 1.5 $          | $70.2 \pm 1.5 $         |
| DGCNN                      | $69.2 \pm 3.0 $         | $45.6 \pm 3.4 $          | $87.8 \pm 2.5 $         | $49.2 \pm 1.2 $          | $71.2 \pm 1.9 $         |
| DiffPool                   | $68.4 \pm 3.3 $         | $45.6 \pm 3.4 $          | $89.1 \pm 1.6 $         | $53.8 \pm 1.4 $          | $68.9 \pm 2.0 $         |
| ECC                        | $67.7 \pm 2.8 $         | $43.5 \pm 3.1 $          | -                       | -                        | -                       |
| GIN                        | $71.2 \pm 3.9 $         | $48.5 \pm 3.3 $          | $89.9 \pm 1.9 $         | $\mathbf{56.1} \pm 1.7 $ | $75.6 \pm 2.3 $         |
| GraphSAGE                  | $68.8 \pm 4.5 $         | $47.6 \pm 3.5 $          | $84.3 \pm 1.9 $         | $50.0 \pm 1.3 $          | $73.9 \pm 1.7 $         |
| [CGMM](https://www.jmlr.org/papers/volume21/19-470/19-470.pdf)                       | $\mathbf{72.7} \pm 3.6$          | $47.5 \pm 3.9$           | $88.1 \pm 1.9$          | $52.4 \pm 2.2$           | $77.32 \pm 2.2$         |
| [ECGMM](https://ieeexplore.ieee.org/document/9533430/)                      | $70.7 \pm 3.8$          | $48.3 \pm 4.1 $          | $89.5 \pm 1.3$          | $53.7 \pm 1.0 $          | $77.45 \pm 2.3$         |
| [iCGMM<sub>*f*</sub>](https://proceedings.mlr.press/v162/castellana22a/castellana22a.pdf)                | $71.8 \pm 4.4$          | $49.0 \pm 3.8 $          | $\mathbf{91.6} \pm 2.1$ | $55.6 \pm 1.7$           | $\mathbf{78.9} \pm 1.7$ |
| [GSPN](https://arxiv.org/pdf/2305.10544.pdf) | - | - | $90.5 \pm 1.1$ | $55.3 \pm 2.0$ |$78.1 \pm 2.5$ |


#### If you are interested in an introduction to Deep Graph Networks **(and a new library!)**, check this out:

[Bacciu Davide, Errica Federico, Micheli Alessio, Podda Marco: *A Gentle Introduction to Deep Learning for Graphs*](https://arxiv.org/abs/1912.12693), Neural Networks, 2020. DOI: `10.1016/j.neunet.2020.06.006`.


### Installation

We provide a script to install the environment. You will need the conda package manager, which can be installed from [here](https://www.anaconda.com/products/individual).

To install the required packages, follow there instructions (tested on a linux terminal):

1) clone the repository

    git clone https://github.com/diningphil/gnn-comparison

2) `cd` into the cloned directory

    cd gnn-comparison

3) run the install script

    source install.sh [<your_cuda_version>]

Where `<your_cuda_version>` is an optional argument that can be either `cpu`, `cu92`, `cu100`, `cu101`. If you do not provide a cuda version, the script will default to `cpu`. The script will create a virtual environment named `gnn-comparison`, with all the required packages needed to run our code. **Important:** do NOT run this command using `bash` instead of `source`!


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

`cp -r DATA/[CHEMICAL|SOCIAL_1|SOCIAL_DEGREE]/<name> DATA` \
`python Launch_Experiments.py --config-file <config> --dataset-name <name> --result-folder <your-result-folder> --debug`

Where `<config>` is your config file (e.g. config_BaselineChemical.yml), and `<name>` is the dataset name chosen as before.

### Additional Notes

You can only use CUDA with the `--debug` option, parallel GPUs support is not provided.

### Troubleshooting

<!-- The installation of Pytorch Geometric depends on other libraries (torch_scatter, torch_cluster, torch_sparse) that have to be installed separately and before torch_geometric. Do not use pip install -r requirements.txt because it will not work. Please refer to the [official instructions](https://github.com/rusty1s/pytorch_geometric) to install the required libraries. -->

If you would like PyTorch not to spawn multiple threads for each process (**highly recommended**), append ``export OMP_NUM_THREADS=1`` to your .bashrc file.

