import torch
from cvxopt import matrix, spdiag, solvers
import numpy as np
from models import FocalLoss
import time

def compute_weights(lossgrad, lamb):

    device = lossgrad.get_device()
    lossgrad = lossgrad.data.cpu().numpy()

    # Compute Optimal sample Weights
    aux = -(lossgrad**2+lamb)
    sz = len(lossgrad)
    P = 2*matrix(lamb*np.identity(sz))
    q = matrix(aux.astype(np.double))
    A = spdiag(matrix(-1.0, (1,sz)))
    b = matrix(0.0, (sz,1))
    Aeq = matrix(1.0, (1,sz))
    beq = matrix(1.0*sz)
    initvals = b+1
    solvers.options['show_progress'] = False
    solvers.options['maxiters'] = 20
    solvers.options['abstol'] = 1e-4
    solvers.options['reltol'] = 1e-4
    solvers.options['feastol'] = 1e-4
    sol=solvers.qp(P, q, A, b, Aeq, beq)
    w = np.array(sol['x'])
    
    return torch.squeeze(torch.tensor(w,dtype=torch.float,device=device))

def LOWLoss(output, target, lamb=1, loss='CE', classprobs=None):
    # higher lamb means more smoothness - weights closer to 1
    
    device = output.get_device()
    
    
    start = time.time()
    
    if loss=='CE':
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        
    elif loss=='BCE':
        if classprobs==None:
            print('please define classprobs')
            classprobs = torch.ones(output.shape[1],device=device)/output.shape[1]
        loss_fn = torch.nn.CrossEntropyLoss(weight=classprobs, reduction='none')
        
    elif loss=='FL':
        loss_fn = FocalLoss(num_class=output.shape[1], gamma=2.0, alpha=0.25, smooth=1e-4)

#    # Compute loss gradient norm (Slow implementation based on autograd)
#    lossgrad = torch.zeros(output.shape[0],device=device)
#    for i in range(output.shape[0]):
#        target_i = target[i].view(1)
#        output_d = output[i].detach().view(1,-1)
#        output_d.requires_grad_(True)
#        loss_di = loss_fn(output_d,target_i)
#        loss_di.backward()
#        lossgrad[i] = torch.norm(output_d.grad,2,1)

#    # Compute cross-entropy gradient (Fast implementation)
#    prob = torch.nn.Softmax(dim=1)(output)
#    labels = torch.t(torch.unsqueeze(target,0))
#    onehot = torch.zeros(prob.size(),device=device)
#    onehot = onehot.scatter_(1, labels, 1)
#    lossgrad = torch.norm(torch.add(prob,-onehot),2,1)
        
    # Compute loss gradient norm (Fast implementation based on autograd)
    output_d = output.detach()
    loss_d = loss_fn(output_d.requires_grad_(True),target)
    loss_d.backward(torch.ones_like(loss_d))
    lossgrad = torch.norm(output_d.grad,2,1)
    
    # Computed weighted loss
    weights = compute_weights(lossgrad, lamb)
    loss = torch.nn.functional.cross_entropy(output, target,reduction='none')
    loss_ww = torch.mean(loss,dim=0)
    loss = torch.mean(torch.mul(loss,weights),dim=0)
    
    print(time.time() - start)
    
    return loss, weights, loss_ww