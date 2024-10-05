# Sim2HW: Modeling Latency Offset Between Network Simulations and Hardware Measurements

This repository contains the dataset and instructions to reproduce the results presented in the paper *Sim2HW: Modeling Latency Offset Between Network Simulations and Hardware Measurements* by Johannes Späth, Max Helm, Benedikt Jaeger, and Georg Carle.
The paper was published at the [GNNet '24 Workshop](https://bnn.upc.edu/workshops/gnnet2024/).

If you find our work useful, please consider citing it:

```bibtex
@inproceedings{spaeth2024sim2hw,
  title = {{Sim2HW: Modeling Latency Offset Between Network Simulations and Hardware Measurements}},
  author = {Späth, Johannes and Helm, Max and Jaeger, Benedikt and Carle, Georg},
  booktitle = {Proceedings of the 3rd GNNet Workshop: Graph Neural Networking Workshop (GNNet '24)},
  address = {Los Angeles, CA, USA},
  isbn = {979-8-4007-1254-8/24/12},
  publisher = {Association for Computing Machinery},
  doi = {10.1145/3694811.3697820},
  year = 2024,
  month = dec
}
```


## Network Configurations
The network configurations, containing the different topologies and flow configurations, are located in `network-configs/`.


## HVNet Dataset
> [!WARNING]
> This step will download and store around 280 GB of data on the local disk.

The HVNet dataset by Florian Wiedner et al. [1] can be downloaded into `latency-data/hvnet/` and preprocessed by using the following script:
```bash
pushd latency-data/hvnet
./download_hvnet_data.sh
popd
```

The sending behavior of the flows in the HVNet dataset is modeled by a gamma distribution for the simulation.
The gamma parameters are located in `hvnet-flow-gamma-params/gamma-params.json` and can be obtained with the Jupyter Notebook in `hvnet-flow-gamma-params/extract_gamma-params.ipynb`.


## OMNeT++ Dataset
> [!WARNING]
> OMNeT++ will generate lots of data, the final latency files for all topologies are around 100 GB.

The OMNeT++ configurations and run scripts for the different topologies are located in `omnet-configs/`.
Before running the simulations, make sure to install the prerequisites:
```bash
pushd omnet-configs
./prepare_machine.sh
popd
```

Afterwards, the individual simulations can be run using the `run.sh` scripts within `omnet-configs/nw_<X>/`.
This will also preprocess the latency data and store it in `latency-data/omnet/`.


## Latency Distributions and Correlation Between HVNet and OMNeT++
Once both the HVNet and OMNeT++ latency files are in `latency-data/`, the latency distributions correlation between the two datasets can be analyzed.
This can be done with the following script:
```bash
pushd latency-analysis
python3 analyze_latency_data.py
popd
```

This will create individual files for each topology and combine them into the following 3 files in `latency-analysis/results/`:
* `latency-stats-hvnet.csv`: stochastic properties of the latency values obtained for each topology with HVNet
* `latency-stats-omnet.csv`: stochastic properties of the latency values obtained for each topology with OMNeT++
* `latency-corrcoefs.csv`: correlation coefficients for the stochastic properties of HVNet and OMNeT++ latency values


## GNN
The GNN models and scripts are located in `gnn/`.
To execute the pipeline, make sure to have the NVIDIA CUDA driver for your GPU installed and install the prerequisites using the script:
```bash
pushd gnn
./prepare-machine.sh
```
Note that all GNN scripts need to be executed in `gnn/`.

### Preprocessing
The datasets need to be preprocessed into a graph representation for the GNN:
```bash
python3 preprocess.py
```

This yields two files in `gnn/data/input-datasets/predict-quantiles-log-normalize/`:
* `training/train.npz`: training dataset
* `test.npz`: test dataset


### Training
The GNN is trained using the following command:
```bash
python3 neural_network.py --mape --regression --device cuda \
    --dataset data/input-datasets/predict-quantiles-log-normalize/training/ \
    --epochs 100 --batch-size 2 --train-test-split 0.85 \
    --learning-rate 0.005405828481669532 --lr-scheduler-factor 0.7 \
    --dropout 0.00017346535607528496 --dropout-gru 0.22756250899600572 \
    --hidden-size 32 --nunroll 4 --num-layers 1 \
    --linear-layer-input --model-architecture SAGE
```

Note that we already set the hyper-parameters to the values obtained by the optimization process described below.
For more information on the available command-line parameters, issue the `neural_network.py` script with the `--help` flag.


### Hyper-Parameter Optimization
To start the hyper-parameter optimization with the [NNI tool](https://nni.readthedocs.io/en/stable/), run the following command:
```bash
nnictl create --config nni-configs/nni-config-om2hvnet-hpo.yml
```

This will start NNI in the background and show a URL for the webinterface.
After the optimization process, the hyper-parameters for the different trials can be seen in the webinterface.
We provide our optimized parameters in `gnn/results/nni-hpo-params-best-trial.json`.

Further, NNI will store the epochs of the trials in `gnn/data/output-models/nni/`.
We copied the data of the best trial into `gnn/data/output-models/predict-quantiles-log-normalize/`.
Note that the best performing epoch was `model_epoch_0090.out`.


### Prediction
To run the predictions on the test dataset, execute the following command:
```bash
python3 predict.py \
    --input-model data/output-models/predict-quantiles-log-normalize/model_epoch_0090.out \
    --dataset data/input-datasets/predict-quantiles-log-normalize/test.npz
```

This yields the file `gnn/results/rel_errs.csv`, which contains the relative errors of the predictions.


### Feature Importance
To determine the feature importance, run the following commands:
```bash
# create feature importance parameter files
python3 export_feature_importance_params.py

# run feature importance script
python3 feature_importance.py \
    --input-model data/output-models/predict-quantiles-log-normalize/model_epoch_0090.out \
    --dataset data/input-datasets/predict-quantiles-log-normalize/test.npz \
    --columns feature-importance-params/sim2hw_feature_importance_params_aggregated.yml \
    --output results/feature_importance.csv --num-epochs 40
```

This creates the file `results/feature_importance.csv`, which contains the feature importance data.


## Plotting
The data can be visualized by running the Jupyter Notebook in `plotting/sim2hw-paper-plots.ipynb`.
This stores the plots in `plotting/plots/`.


## References
[1]  Florian Wiedner, Max Helm, Sebastian Gallenmüller, and Georg Carle. 2022. HVNet: Hardware-Assisted Virtual Networking on a Single Physical Host. https://mediatum.ub.tum.de/1638129
