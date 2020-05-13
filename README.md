# Deep Neural Network (DNN) to Support Vector Machine (SVM)

*Inspired by DLSVM { Y. Tang's [Deep Learning using Linear Support Vector Machines](https://arxiv.org/abs/1306.0239) (2013) }*

## Requirements
- [Docker](https://www.docker.com/) >= 19.03
- [GNU Make](https://www.gnu.org/software/make/)
- [nvidia-drivers](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#how-do-i-install-the-nvidia-driver) (Only for GPU)
- [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) (Only for GPU)

## Installation
Before installation, you must meet requirements.

```bash
$ git clone https://github.com/s-sakagawa/dnn-svm
$ cd dnn-svm
$ make build
```

## Implemented Commands
The commands are implemented by Makefile.

### Run Jupyter Lab
```bash
$ make lab
```

### Run bash
```bash
$ make bash
```

If you want to run bash by `root`, run below.
```bash
$ make bash USER=root
```
This command is useful to add some packages by poetry.
