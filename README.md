# Simple example of IW-SGD
A PyTorch implementation of IW-SGD on MNIST with LeNet-5.

## Recent updates
1. Original submission

## Motivation
IW-SGD is a new learning strategy that applies specific weights to training samples according to their gradient norm (inspired by importance sampling)

## Requirements
- PyTorch
- CUDA

## Usage

**Running the demo:**

- IW-SGD:

```sh
python demo.py --data <path_to_folder_with_mnist> --save <path_to_save_dir> --alpha=0.9
```

- Standard SGD:

```sh
python demo.py --data <path_to_folder_with_mnist> --save <path_to_save_dir> --alpha=0
```

Options:
- `--alpha` (float) - hyperparameter of IW-SGD (between 0 and 1) (default 0.9)
- `--n_epochs` (int) - number of epochs for training (default 100)
- `--batch_size` (int) - size of minibatch (default 256)
- `--seed` (int) - manually set the random seed (default None)

## Performance

A comparison between SGD and IW-SGD:

|    alpha    | Test error (%) |
|-------------|----------------|
| 0 (SGD)     |     2.190      |
| 1           |     1.440      |

## Reference

```
@article{unpublished,
  title={Training Deep Neural Networks Efficiently with Importance Weighted Stochastic Gradient Descent},
  year={2019}
}
```
