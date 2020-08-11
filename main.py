import os
import time
import numpy as np
import torch
from torchvision import transforms, datasets
import argparse
from lenet import LeNet5
from LOW import LOWLoss

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def train_epoch(model, loader, optimizer, epoch, n_epochs, loss_func, lamb):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    # Loss function
    if loss_func == 'CE':
        loss_fn = torch.nn.CrossEntropyLoss()
    elif loss_func == 'LOW':
        loss_fn = LOWLoss(lamb=lamb)

    # Model on train mode
    model.train()

    end = time.time()
    for batch_idx, (input, target) in enumerate(loader):
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # compute output
        output = model(input)

        # compute loss
        loss = loss_fn(output, target)

        # measure accuracy and record loss
        batch_size = target.size(0)
        _, pred = output.data.cpu().topk(1, dim=1)
        error.update(torch.ne(pred.squeeze(), target.cpu()).float().sum() / batch_size, batch_size)
        losses.update(loss.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
                
        # print stats
        res = '\t'.join([
            'Epoch: [%d/%d]' % (epoch + 1, n_epochs),
            'Iter: [%d/%d]' % (batch_idx, len(loader)),
            'Time %.3f (%.3f)' % (batch_time.sum, batch_time.avg),
            'Loss %.4f' % losses.avg,
            'Error %.4f' % error.avg,
        ])
        print(res)

    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg


def test_epoch(model, loader):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    # Model on eval mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = torch.nn.functional.cross_entropy(output, target)
      
            # measure accuracy and record loss
            batch_size = target.size(0)
            _, pred = output.data.cpu().topk(1, dim=1)
            error.update(torch.ne(pred.squeeze(), target.cpu()).float().sum() / batch_size, batch_size)
            losses.update(loss.item(), batch_size)
      
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print stats
            res = '\t'.join([
                'Test',
                'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                'Time %.3f (%.3f)' % (batch_time.sum, batch_time.avg),
                'Loss %.4f' % losses.avg,
                'Error %.4f' % error.avg,
            ])
            print(res)

    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg


def train(model, train_loader, test_loader, save, loss_func, lr=0.001, n_epochs=100, lamb=0.1):

    # Model on cuda
    if torch.cuda.is_available():
        model = model.cuda()

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0001)

    # Start log
    with open(os.path.join(save, 'results.csv'), 'w') as f:
        f.write('epoch,train_loss,train_error,test_loss,test_error\n')
        
    # Train model
    for epoch in range(n_epochs):
        _, train_loss, train_error = train_epoch(
            model,
            train_loader,
            optimizer,
            epoch,
            n_epochs,
            loss_func,
            lamb,
        )
        _, test_loss, test_error = test_epoch(
            model,
            test_loader,
        )

        torch.save(model.state_dict(), os.path.join(save, 'model.dat'))

        # Log results
        with open(os.path.join(save, 'results.csv'), 'a') as f:
            f.write('%03d,%0.6f,%0.6f,%0.5f,%0.5f,\n' % (
                (epoch + 1),
                train_loss,
                train_error,
                test_loss,
                test_error,
            ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/tmp/', help='location of MNIST dataset')
    parser.add_argument('--save', type=str, default='./save/', help='location of results logs')
    parser.add_argument('--loss_func', type=str, default='LOW', help="loss function - eg 'CE' or 'LOW' (default)")
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lamb', type=float, default=0.1, help='lambda parameter in LOW')

    # Setup
    opt = parser.parse_args()
    data = opt.data
    loss_func = opt.loss_func
    save = os.path.join(opt.save, loss_func)
    n_epochs = opt.n_epochs
    batch_size = opt.batch_size
    lr = opt.lr
    lamb = opt.lamb

    # Dataset and data transformations
    train_set = datasets.MNIST(data, train=True, transform=transforms.ToTensor(), download=True)
    test_set = datasets.MNIST(data, train=False, transform=transforms.ToTensor(), download=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    n_classes = 10
    n_channels = 1
    input_size = 28

    # Model
    model = LeNet5(n_channels=n_channels, inputsize=input_size, outputsize=n_classes)

    # Make save directory
    if not os.path.exists(save):
        os.makedirs(save)
    if not os.path.isdir(save):
        raise Exception('%s is not a dir' % save)

    # Train the model
    train(model, train_loader, test_loader, save, loss_func, lr=lr, n_epochs=n_epochs, lamb=lamb)



