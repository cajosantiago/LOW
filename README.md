# Official Pytorch implementation of "LOW: Training Deep Neural Networks by Learning Optimal Sample Weights"
A PyTorch implementation of LOW from the paper "LOW: Training Deep Neural Networks by Learning Optimal Sample Weights" and an example on MNIST with LeNet-5.

## Recent updates
1. Original submission

## Requirements
- PyTorch
- cvxopt
- numpy

## Usage

**Running the demo:**

- LOW on MNIST:

```sh
python main.py --data <path_to_folder_with_mnist> --save <path_to_save_dir> --loss 'LOW'
```

- Standard SGD on MNIST:

```sh
python main.py --data <path_to_folder_with_mnist> --save <path_to_save_dir> --loss 'SGD'
```

Options:
- `--loss` (str) - available losses: 'SGD' and 'LOW' (default 'LOW')
- `--n_epochs` (int) - number of epochs for training (default 100)
- `--batch_size` (int) - size of minibatch (default 256)

## Performance

A comparison between SGD and LOW:

|    Loss     | Test error (%) |
|-------------|----------------|
|     SGD     |     1.74       |
|     LOW     |     0.85       |

## Reference

```
@article{unpublished,
  title={LOW: Training Deep Neural Networks by Learning Optimal Sample Weights},
  year={2020}
}
```
