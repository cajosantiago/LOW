import torch
from cvxopt import matrix, spdiag, solvers
import numpy as np

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
    solvers.options['show_progress'] = False
    solvers.options['maxiters'] = 20
    solvers.options['abstol'] = 1e-4
    solvers.options['reltol'] = 1e-4
    solvers.options['feastol'] = 1e-4
    sol = solvers.qp(P, q, A, b, Aeq, beq)
    w = np.array(sol['x'])
    
    return torch.squeeze(torch.tensor(w, dtype=torch.float, device=device))

class LOWLoss(torch.nn.Module):
    def __init__(self, lamb=0.1):
        super(LOWLoss, self).__init__()
        self.lamb = lamb # higher lamb means more smoothness -> weights closer to 1
        self.loss = torch.nn.CrossEntropyLoss(reduction='none')  # replace this with any loss with "reduction='none'"

    def forward(self, logits, target):
        # Compute loss gradient norm
        output_d = logits.detach()
        loss_d = torch.mean(self.loss(output_d.requires_grad_(True), target), dim=0)
        loss_d.backward(torch.ones_like(loss_d))
        lossgrad = torch.norm(output_d.grad, 2, 1)

        # Computed weighted loss
        weights = compute_weights(lossgrad, self.lamb)
        loss = self.loss(logits, target)
        loss = torch.mean(torch.mul(loss, weights), dim=0)

        return loss