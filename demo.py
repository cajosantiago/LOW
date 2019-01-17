import fire
import os
import time
import torch
from torchvision import datasets, transforms
import numpy


class LeNet5(torch.nn.Module):   
    """
    Copied from: https://github.com/bollakarthikeya/LeNet-5-PyTorch/blob/master/lenet5_gpu.py
    """       
     
    def __init__(self):     
        super(LeNet5, self).__init__()
        # Convolution (In LeNet-5, 32x32 images are given as input. Hence padding of 2 is done below)
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        # Max-pooling
        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2)
        # Convolution
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        # Max-pooling
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2) 
        # Fully connected layer
        self.fc1 = torch.nn.Linear(16*5*5, 120)   # convert matrix with 16*5*5 (= 400) features to a matrix of 120 features (columns)
        self.fc2 = torch.nn.Linear(120, 84)       # convert matrix with 120 features to a matrix of 84 features (columns)
        self.fc3 = torch.nn.Linear(84, 10)        # convert matrix with 84 features to a matrix of 10 features (columns)
        
    def forward(self, x):
        # convolve, then perform ReLU non-linearity
        x = torch.nn.functional.relu(self.conv1(x))  
        # max-pooling with 2x2 grid 
        x = self.max_pool_1(x) 
        # convolve, then perform ReLU non-linearity
        x = torch.nn.functional.relu(self.conv2(x))
        # max-pooling with 2x2 grid
        x = self.max_pool_2(x)
        # first flatten 'max_pool_2_out' to contain 16*5*5 columns
        # read through https://stackoverflow.com/a/42482819/7551231
        x = x.view(-1, 16*5*5)
        # FC-1, then perform ReLU non-linearity
        x = torch.nn.functional.relu(self.fc1(x))
        # FC-2, then perform ReLU non-linearity
        x = torch.nn.functional.relu(self.fc2(x))
        # FC-3
        x = self.fc3(x)
        
        return x



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

        
def train_epoch(model, loader, optimizer, epoch, n_epochs, alpha=0):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    # Model on train mode
    model.train()

    end = time.time()
    for batch_idx, (input, target) in enumerate(loader):
        # Create vaiables
        if torch.cuda.is_available():
            input_var = torch.autograd.Variable(input.cuda(async=True))
            target_var = torch.autograd.Variable(target.cuda(async=True))
        else:
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        
        if alpha>0:
            ############################ IW-SGD ##############################
            softmax = torch.nn.Softmax(dim=1) 
            prob = softmax(output).data
            labels = torch.t(torch.unsqueeze(target_var,0)).data
            onehot = torch.zeros(prob.size(),device=labels.get_device())
            onehot = onehot.scatter_(1, labels, 1)
            aux = torch.norm(torch.add(prob,-onehot),2,1).data
            weights = alpha*torch.exp(aux).data + (1-alpha)*1 
            loss = torch.nn.functional.cross_entropy(output, target_var,reduction='none')
            loss = torch.mean(torch.mul(loss,weights),dim=0)
            ############################ IW-SGD ##############################
        else:
            ############################ SGD ##############################
            loss = torch.nn.functional.cross_entropy(output, target_var)
            ############################ SGD ##############################

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
        'Time %.3f' % (batch_time.avg*len(loader)),
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
    for batch_idx, (input, target) in enumerate(loader):
        # Create vaiables
        if torch.cuda.is_available():
            input_var = torch.autograd.Variable(input.cuda(async=True), volatile=True)
            target_var = torch.autograd.Variable(target.cuda(async=True), volatile=True)
        else:
            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = torch.nn.functional.cross_entropy(output, target_var)

        # measure accuracy and record loss
        batch_size = target.size(0)
        _, pred = output.data.cpu().topk(1, dim=1)
        error.update(torch.ne(pred.squeeze(), target.cpu()).float().sum() / batch_size, batch_size)
        losses.update(loss.data[0], batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    # print stats
    res = '\t'.join([
        'Test',
        'Time %.3f' % (batch_time.avg*len(loader)),
        'Loss %.4f' % losses.avg,
        'Error %.4f' % error.avg,
    ])
    print(res)

    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg


def train(model, train_set, test_set, save, n_epochs=100, 
          batch_size=256, lr=0.0001, wd=0.0001, momentum=0.9, seed=None, alpha=.9): 
    if seed is not None:
        torch.manual_seed(seed)

    # Data loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                              pin_memory=(torch.cuda.is_available()), num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                              pin_memory=(torch.cuda.is_available()), num_workers=0)

    # Model on cuda
    if torch.cuda.is_available():
        model = model.cuda()

    # Wrap model for multi-GPUs, if necessary
    model_wrapper = model
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model_wrapper = torch.nn.DataParallel(model).cuda()

    # Optimizer
    optimizer = torch.optim.SGD(model_wrapper.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=wd)

    # Start log
    with open(os.path.join(save, 'results.csv'), 'w') as f:
        f.write('epoch,train_loss,train_error,test_loss,test_error\n')
        
    # Train model
    best_error = 1
    for epoch in range(n_epochs):
        #scheduler.step()
        _, train_loss, train_error = train_epoch(
            model=model_wrapper,
            loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs,
            alpha=alpha,
        )
        _, test_loss, test_error = test_epoch(
            model=model_wrapper,
            loader=test_loader,
        )
        
        # Save model
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

#    # Final test of model on test set
#    model.load_state_dict(torch.load(os.path.join(save, 'model.dat')))
#    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
#        model = torch.nn.DataParallel(model).cuda()
#    test_results = test_epoch(
#        model=model,
#        loader=test_loader,
#        is_test=True
#    )
#    _, _, test_error = test_results
#    with open(os.path.join(save, 'results.csv'), 'a') as f:
#        f.write(',,,,,%0.5f\n' % (test_error))
    print('Final test error: %.4f' % test_error)


def demo(data, save, n_epochs=100, batch_size=256, seed=None, alpha=0.9):
    """
    A demo to show off training using IW-SGD.
    Trains and evaluates a LeNet-5 on MNIST.

    Args:
        data (str) - path to directory where data should be loaded from/downloaded
            (default $DATA_DIR)
        save (str) - path to save the model to (default /tmp)

        n_epochs (int) - number of epochs for training (default 300)
        batch_size (int) - size of minibatch (default 256)
        seed (int) - manually set the random seed (default None)
        
        alpha (float) - hyperparameter of IW-SGD (between 0 and 1) (default 0.9)
    """

    # Data transforms
    mean = [0.5, 0.5, 0.5]
    stdv = [0.5, 0.5, 0.5]
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])

    # Datasets
    train_set = datasets.MNIST(data, train=True, transform=train_transforms, download=True)
    test_set = datasets.MNIST(data, train=False, transform=test_transforms, download=True)
    

    model = LeNet5()
    
    print(model)
        
    # Make save directory
    if not os.path.exists(save):
        os.makedirs(save)
    if not os.path.isdir(save):
        raise Exception('%s is not a dir' % save)

    # Train the model
    train(model=model, train_set=train_set, test_set=test_set, save=save,
          n_epochs=n_epochs, batch_size=batch_size, seed=seed, alpha=alpha)
    print('Done!')

"""
A demo to show off training using IW-SGD.
Trains and evaluates a LeNet-5 on MNIST.

Args:
    data (str) - path to directory where data should be loaded from/downloaded
        (default $DATA_DIR)
    save (str) - path to save the model to (default /tmp)

    n_epochs (int) - number of epochs for training (default 300)
    batch_size (int) - size of minibatch (default 256)
    seed (int) - manually set the random seed (default None)
    
    alpha (float) - hyperparameter of IW-SGD (between 0 and 1) (default 0.9)
"""

if __name__ == '__main__':
    fire.Fire(demo)
