# v1.0 Release Note

"regularizeCNN" provides essential tools for evaluating the effects of different regularizers/optimizers when training a neural network.

## Structure

![Module diagram](./structure.svg  "Module diagram")

## System requirements

### Hardware (recommended)

* i7-950 or equivalent
* 8G+ DRAM
* Nvidia Geforce GPU with CUDA capability. 8G graphics memory.

### Hardware (only for small batch evaluation)

* i7-950 or equivalent
* 8G+ DRAM

### OS/Build Environment

\*\* Listed is my working environment. Some other may also work.

* Win7/8/10 /Ubuntu 18.04+/ Debian 9.7.0+
* Python 3.7
* Pytorch 1.5.0 + torchvision 0.6.0
* numpy 1.18.5
* matplotlib 3.2.2
* tqdm 4.46.1

## Usage

**demoTrain.py** : Train a new neural-net with specified dataset and save to file "._outputs/\{timestamp\}.model".

**demoAdversary.py** : Load a pre-trained net from file and generate adversary samples that can fool the net.
