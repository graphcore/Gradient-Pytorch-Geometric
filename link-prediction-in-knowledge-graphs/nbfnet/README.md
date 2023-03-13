# NBFNet: Neural Bellman-Ford Networks
This directory contains a PyTorch Geometric implementation of [NBFNet](https://arxiv.org/abs/2106.06935) for link
prediction in homogeneous and heterogeneous graphs, optimised for Graphcore IPUs.

Run inductive training on the FB15k-237 knowledge graph  on Paperspace.
<br>
[![Gradient](../../../gradient-badge.svg)](TODO)

| Framework | Domain | Model | Datasets | Tasks | Training | Inference | Reference |
|-----------|--------|-------|----------|-------|----------|-----------|-----------|
| PyTorch | GNNs | NBFNet | FB15k-237 | Link Prediction | <div style="text-align: center;">✅ <br>Min. 4 IPUs (POD4) required | <p style="text-align: center;">❌ | [Neural Bellman-Ford Networks: A General Graph Neural Network Framework for Link Prediction](https://arxiv.org/abs/2106.06935) |

## Instructions summary

1. Install and enable the Poplar SDK (see Poplar SDK setup)

2. Install the system and Python requirements (see Environment setup)

3. Run the Jupyter notebook or train the model directly from the terminal (see Running)

## Poplar SDK setup

To check if your Poplar SDK has already been enabled, run:
```bash
 echo $POPLAR_SDK_ENABLED
 ```

If no path is provided, then follow these steps:

1. Navigate to your Poplar SDK root directory
2. Enable the Poplar SDK with:
```bash
cd poplar-<OS version>-<SDK version>-<hash>
. enable.sh
```

3. Additionally, enable PopART with:
```bash
cd popart-<OS version>-<SDK version>-<hash>
. enable.sh
```

More detailed instructions on setting up your Poplar environment are available in the [Poplar quick start guide](https://docs.graphcore.ai/projects/poplar-quick-start).


## Environment setup

To prepare your environment, follow these steps:

1. Create and activate a Python3 virtual environment:
```bash
python3 -m venv <venv name>
source <venv path>/bin/activate
```

2. Navigate to the Poplar SDK root directory

3. Install the PopTorch (PyTorch) wheel:
```bash
cd <poplar sdk root dir>
pip3 install poptorch...x86_64.whl
```

4. Install the PopTorch Geometric wheel:
```bash
cd <poplar sdk root dir>
pip3 install poptorch_geometric....whl
```

5. Navigate to this example's root directory

6. Install the Python requirements:
```bash
pip3 install -r requirements.txt
```

More detailed instructions on setting up your PyTorch environment are available in the [PyTorch quick start guide](https://docs.graphcore.ai/projects/pytorch-quick-start).


## Running

The model can be trained in an interactive in a jupyter notebook using the `NBFNet_training.ipynb` notebook.

To train an inductive model on the FB15k-237 dataset from your terminal run
```bash
python run_nfbnet.py -c configs/IndFB15k-237_v1.yaml
```
The `dataset.version` parameter in the config file determines the data split (allowed values: `["v1", "v2", "v3", "v4"]`).

Model specific hyperparameters can be set in the `.yaml` config file. Additionally, the following flags can be passed to
`run_nfbnet.py` to adjust its behaviour:

-c, --config: Path to a config file <br>
--device: The device to execute on ("ipu" or "cpu") <br>
--profile: Generate a popvision profile <br>
--profile_dir: Directory to save popvision profile <br>


## License
This application is licensed under the MIT license, see the LICENSE file at the top-level of this repository.

This directory includes derived work from https://github.com/KiddoZhu/NBFNet-PyG  <br>
Copyright (c) 2021 MilaGraph <br>
Licensed under the MIT License
