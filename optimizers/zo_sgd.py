import torch
from torch.optim import Optimizer


class ZO_SGD(Optimizer):
    def __init__(self, params, lr=1e-03, mu =1e-03):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: (} - should be >= 0.0".format(lr))

        defaults = dict(lr=lr, mu = mu)
        super().__init__(params, defaults)
        # Compute the size of the parameters vector
        self.size_params = 0
        for group in self.param_groups:
            for p in group['params']:
                self.size_params += torch.numel(p)

    def step(self, closure):
        for group in self.param_groups:
            # closure return the approximation for the gradient, we have to add some "option" to this function
            grad_est = closure(self.size_params, group["mu"])

            for p, grad in zip(group['params'], grad_est):
                p.data.add_(grad, alpha=-group["lr"])
