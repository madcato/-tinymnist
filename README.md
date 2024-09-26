# tinymnist

MNIST implementation using [tinygrad](https://github.com/tinygrad/tinygrad).

Based on this pytorch tutorial: [MNIST Handwritten Digit Recognition in PyTorch](https://nextjournal.com/gkoehler/pytorch-mnist) and this one [Aplicación de una red neuronal convolucional en un conjunto de datos mnist](https://www.geeksforgeeks.org/applying-convolutional-neural-network-on-mnist-dataset/)

- [Benchmark MNIST models](https://paperswithcode.com/sota/image-classification-on-mnist)
- Model TinyMNISTNRNBCNet tries to reproduce this paper [NO ROUTING NEEDED BETWEEN CAPSULES](https://arxiv.org/pdf/2001.09136v6)


## Install tinygrad

Clone the repo, in a different directory than this project one:
```bash
git clone https://github.com/tinygrad/tinygrad.git
cd tinygrad
python3 -m pip install -e .
```

## Run

```bash
python3 simple-sample.py
```
