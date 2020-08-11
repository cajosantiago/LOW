# Official Pytorch implementation of "LOW: Training Deep Neural Networks by Learning Optimal Sample Weights"
A PyTorch implementation of "LOW: Training Deep Neural Networks by Learning Optimal Sample Weights" with an example on MNIST using LeNet-5.

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

- Standard CE on MNIST:

```sh
python main.py --data <path_to_folder_with_mnist> --save <path_to_save_dir> --loss 'CE'
```

Options:
- `--loss` (str) - available losses: 'CE' and 'LOW' (default)
- `--n_epochs` (int) - number of epochs (default 100)
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
  author={Santiago, C. and Barata, C. and Sasdelli, M. and Carneiro, G. and Nascimento, J.},
  journal={Pattern Recognition},
  pages={1--1},
  year={2020},
  publisher={Elsevier}
}
```
