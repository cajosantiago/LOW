#import fire
import os
import time
import numpy as np
import torch
import datetime
from torchvision import transforms, datasets
from models import LeNet5,FocalLoss,DenseNet,Wide_ResNet,WideResNet
from cvxopt import matrix, spdiag, solvers
import mnist
import cifar
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

def imbalanced_indices(n_classes, train_set, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        
    NTrnS = len(train_set)
    data = torch.utils.data.DataLoader(train_set, batch_size=NTrnS, shuffle=False, pin_memory=False)
    _,target,_ = next(iter(data))
    N = len(target)
    #NpC = N/n_classes # This is not true! number of samples per class may not be constant
    NpC = np.zeros(n_classes)
    for i in range(len(target)):
        NpC[target[i]]=NpC[target[i]]+1
    print(NpC)
    randclass = np.random.permutation(n_classes)
    selected = np.zeros(N)
    shuffled = np.random.permutation(N)
    count = np.zeros(n_classes)
    for j in range(len(target)):
        sidx = shuffled[j]
        scls = target[sidx]
        if count[scls]<=round(NpC[scls]/(2**randclass[scls])):
            selected[sidx]=1
            count[scls]=count[scls]+1
    print(count)
    print('Total number of training samples:')
    print(int(sum(selected)))
    train_indices = np.where(selected == 1)[0]
    return train_indices
    

def load_model(modeltype, inputsize, n_classes):
    
    # Models and Training Parameters
    if modeltype=='lenet':
        n_epochs=100
        batch_size=128
        lr=0.001
        model = LeNet5(n_channels=n_channels, inputsize=inputsize, outputsize=n_classes)
    elif modeltype=='densenet':
        # Densenet Parameters
        efficient=False
        depth=100
        growth_rate=12
        drop_rate = 0#0.2
        n_epochs=300
        batch_size=64
        lr=0.1
        
        if (depth - 4) % 3:
            raise Exception('Invalid depth')
        block_config = [(depth - 4) // 6 for _ in range(3)]
        model = DenseNet(
            growth_rate=growth_rate,
            block_config=block_config,
            num_classes=n_classes,
            small_inputs=True,
            drop_rate=drop_rate,
            efficient=efficient,
        )
    elif modeltype=='wideresnet':
        depth = 28
        widen_factor = 10
        dropout = 0.3
        n_epochs=200
        batch_size=64
        lr=0.1
        model = Wide_ResNet(depth, widen_factor, dropout, n_classes)
    elif modeltype=='wideresnet2':
        depth=28
        widen=2
        dropout=0.3
        n_epochs=100
        batch_size=128
        lr=0.1
        model = WideResNet(depth, n_classes, widen_factor=widen, dropRate=dropout)
    
    
    return model, n_epochs, batch_size, lr
    

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
        
def train_epoch(model, loader, optimizer, epoch, n_epochs, print_freq=1, LS=0):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()
    
    # Initialize sample weight vector
    sweights = torch.zeros(len(loader.dataset))

    # Model on train mode
    model.train()

    end = time.time()
    for batch_idx, (input, target, index) in enumerate(loader):
    #for batch_idx, (input, target) in enumerate(loader):
        # Create vaiables
        if torch.cuda.is_available():
            input_var = input.cuda()
            target_var = target.cuda()

        # compute output
        output = model(input_var)
        
        batch_size = target.size(0)
        weights = torch.tensor(0)
        # Choose learning strategy LS
        if LS==0: # Normal
            loss = torch.nn.functional.cross_entropy(output, target_var)
            loss_ww = loss
        elif LS==1: # Importance-Weighted
            ##################################################################
            # Weighted-SGD
            softmax = torch.nn.Softmax(dim=1) 
            prob = softmax(output).data#.cpu()
            labels = torch.t(torch.unsqueeze(target_var,0)).data#.cpu()
            onehot = torch.zeros(prob.size(),device=labels.get_device())
            onehot = onehot.scatter_(1, labels, 1)
            aux = torch.norm(torch.add(prob,-onehot),2,1).data
            weights = torch.exp(aux).data #################################### new
            norm_weights = (weights/torch.mean(weights)).data #################################### normalized
            loss = torch.nn.functional.cross_entropy(output, target_var,reduction='none')
            loss = torch.mean(torch.mul(loss,norm_weights),dim=0)
            ##################################################################
#            # Log weights
#            with open('weights_' + str(LS) + '.csv','a') as f:
#                f.write('%f\n' % numpy.mean(weights.cpu().numpy()))
        elif LS==2: # Focal Loss
            FL = FocalLoss(num_class=output.size(1), gamma=2.0, alpha=0.25, smooth=1e-4)#, balance_index=2) alpha .y beatriz
            loss = FL(output, target)
            loss_ww = loss
        elif LS==3: # LOW
            loss, weights, loss_ww = LOWLoss(output,target_var,lamb=1)
        elif LS==4: # ISamp
            print('Not implemented')
        elif LS==5: # SPL
            loss_ww = torch.nn.CrossEntropyLoss(reduction='none')(output, target_var)
            _,indices = torch.sort(loss_ww)
            #selected  = min(batch_size,round(.5*batch_size*(1 + 2*epoch/n_epochs)))
            selected  = int(min(batch_size,round(batch_size*np.exp(.1*epoch)/(1 + np.exp(.1*epoch)))))
            weights = torch.zeros_like(loss_ww).scatter_(0,indices[0:selected],1)
            #weights = (loss_ww<loss_ww.median()/(.95**epoch)).float()
            loss = torch.mul(weights,loss_ww).mean()
            loss_ww = loss_ww.mean()
        elif LS==6: # LOW-FL
            loss, weights, loss_ww = LOWLoss(output,target_var,lamb=1,loss='FL')
        
        # measure accuracy and record loss
        _, pred = output.data.cpu().topk(1, dim=1)
        error.update(torch.ne(pred.squeeze(), target.cpu()).float().sum() / batch_size, batch_size)
        losses.update(loss_ww.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        
        # save sample weights
        sweights[index] = weights.cpu()
        #sweights = torch.tensor(0)
                
    # print stats
    res = '\t'.join([
        'Epoch: [%d/%d]' % (epoch + 1, n_epochs),
        'Iter: [%d]' % len(loader),
        'Time %.3f (%.3f)' % (batch_time.sum, batch_time.avg),
        'Loss %.4f' % losses.avg,
        'Error %.4f' % error.avg,
    ])
    print(res)

    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg, sweights.numpy()


def test_epoch(model, loader, print_freq=1, is_test=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    # Model on eval mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for batch_idx, (input, target,_) in enumerate(loader):
        #for batch_idx, (input, target) in enumerate(loader):
            # Create vaiables
            if torch.cuda.is_available():
                input_var = input.cuda()
                target_var = target.cuda()

            # compute output
            output = model(input_var)
            loss = torch.nn.functional.cross_entropy(output, target_var)
      
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
        'Test' if is_test else 'Valid',
        'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
        'Time %.3f (%.3f)' % (batch_time.sum, batch_time.avg),
        'Loss %.4f' % losses.avg,
        'Error %.4f' % error.avg,
    ])
    print(res)

    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg


def train(model, train_loader, valid_loader, test_loader, save, n_epochs=300,
          batch_size=64, lr=0.001, wd=0.0001, momentum=0.9, seed=None, LS=0, numworkers=0): 
    if seed is not None:
        torch.manual_seed(seed)

    # Model on cuda
    if torch.cuda.is_available():
        model = model.cuda()

    # Wrap model for multi-GPUs, if necessary
    model_wrapper = model
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model_wrapper = torch.nn.DataParallel(model).cuda()

    # Optimizer
    optimizer = torch.optim.SGD(model_wrapper.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=wd)
    
    # Scheduler
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.985)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs],gamma=0.1) #densenet
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.3 * n_epochs, 0.6 * n_epochs, 0.8 * n_epochs],gamma=0.2) #wide-resnet
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.3 * n_epochs, 0.6 * n_epochs, 0.8 * n_epochs],gamma=1) #lenet

    # Start log
    with open(os.path.join(save, 'results.csv'), 'w') as f:
        f.write('epoch,train_loss,train_error,valid_loss,valid_error,test_error\n')
        
    # Train model
    weights = np.zeros((n_epochs,len(train_loader.dataset)))
    for epoch in range(n_epochs):
        scheduler.step()
        _, train_loss, train_error, weights[epoch] = train_epoch(
            model=model_wrapper,
            loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs,
            LS=LS,
        )
        _, valid_loss, valid_error = test_epoch(
            model=model_wrapper,
            loader=valid_loader if valid_loader else test_loader,
            is_test=(not valid_loader)
        )

        torch.save(model.state_dict(), os.path.join(save, 'model.dat'))

        # Log results
        with open(os.path.join(save, 'results.csv'), 'a') as f:
            f.write('%03d,%0.6f,%0.6f,%0.5f,%0.5f,\n' % (
                (epoch + 1),
                train_loss,
                train_error,
                valid_loss,
                valid_error,
            ))
            
    # save weights to file
    np.savetxt(os.path.join(save,'weights.csv'),weights,delimiter=',')

    # Final test of model on test set
    test_results = test_epoch(
        model=model,
        loader=test_loader,
        is_test=True
    )
    _, _, test_error = test_results
    with open(os.path.join(save, 'results.csv'), 'a') as f:
        f.write(',,,,,%0.5f\n' % (test_error))
    print('Final test error: %.4f' % test_error)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Script
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

numworkers = 5

# Setup
dataset    = 'mnist' #'mnist' #'cifar10' #'cifar100'
imbalanced = 0
modeltype  = 'lenet' #lenet #densenet #wideresnet2
n_trials   = 1
LS_types   = ['Normal','IW','FL','LOW','ISamp','SPL','LOW-FL']
LS         = [3]#[0,2,3,5]
saveinit   = False
seedinit   = 0
    
        
# Dataset and data transformations
if dataset=='mnist':
    data='/tmp/' + dataset + '/'
    train_transforms = transforms.Compose([transforms.ToTensor()])
    test_transforms = transforms.Compose([transforms.ToTensor()])
    #train_set = datasets.MNIST(data, train=True, transform=train_transforms, download=True)
    #test_set = datasets.MNIST(data, train=False, transform=test_transforms, download=True)
    train_set = mnist.MNIST(data, train=True, transform=train_transforms, download=True)
    test_set = mnist.MNIST(data, train=False, transform=test_transforms, download=True)
    n_classes = 10
    n_channels = 1
    inputsize  = 28
else:
    data='/tmp/' + dataset + '/'
    mean = [0.491, 0.482, 0.447]
    stdv = [0.247, 0.243, 0.262]
    #mean = [0.5, 0.5, 0.5]
    #stdv = [0.5, 0.5, 0.5]
    n_channels = 3
    inputsize  = 32
    train_transforms = transforms.Compose([
        #transforms.Grayscale(num_output_channels=3),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    test_transforms = transforms.Compose([
        transforms.CenterCrop(32),
        #transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    if dataset=='cifar10':
        #train_set = datasets.CIFAR10(data, train=True, transform=train_transforms, download=True)
        #test_set = datasets.CIFAR10(data, train=False, transform=test_transforms, download=False)
        train_set = cifar.CIFAR10(data, train=True, transform=train_transforms, download=True)
        test_set = cifar.CIFAR10(data, train=False, transform=test_transforms, download=False)
        n_classes = 10
    else:
        #train_set = datasets.CIFAR100(data, train=True, transform=train_transforms, download=True)
        #test_set = datasets.CIFAR100(data, train=False, transform=test_transforms, download=False)
        train_set = cifar.CIFAR100(data, train=True, transform=train_transforms, download=True)
        test_set = cifar.CIFAR100(data, train=False, transform=test_transforms, download=False)
        n_classes = 100


for trial2 in range(n_trials):
    trial = trial2 + 0
    seed = seedinit + trial

    # Models & Training Parameters
    model, n_epochs, batch_size, lr = load_model(modeltype, inputsize, n_classes)
    
    init_name = 'initializations/' + modeltype + '-init_' + dataset + '_trial' + str(trial)
    if saveinit:
        # Save model initialization
        torch.save({'model_state_dict': model.state_dict()},init_name)

    # Data loaders
    if imbalanced==0:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                              pin_memory=(torch.cuda.is_available()), num_workers=numworkers)
    else:
        train_indices = imbalanced_indices(n_classes, train_set,seed=seed)
        imbalanced_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=imbalanced_sampler,
                                  pin_memory=(torch.cuda.is_available()), num_workers=numworkers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                          pin_memory=(torch.cuda.is_available()), num_workers=numworkers)
    valid_loader = test_loader
    
    
    for ls in list(LS):
        if imbalanced==0:
            save='save/' + dataset + '/' + modeltype + '/' + LS_types[ls] + '/' + str(trial) + '/'
        else:
            save='save_imbalanced/' + dataset + '/' + modeltype + '/' + LS_types[ls] + '/' + str(trial) + '/'
        
        print(save)
        
        # Load model initialization
        checkpoint = torch.load(init_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Make save directory
        if not os.path.exists(save):
            os.makedirs(save)
        if not os.path.isdir(save):
            raise Exception('%s is not a dir' % save)
        
        # Train the model
        train(model, train_loader, valid_loader, test_loader, save,
            lr=lr, n_epochs=n_epochs, batch_size=batch_size, seed=seed, LS=ls, numworkers=numworkers)


print('Done!')
print(datetime.datetime.now())


