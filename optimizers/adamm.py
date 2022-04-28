import torch
from torch.optim.optimizer import Optimizer


class AdaMM(Optimizer):
    """
    Implements the first order (FO) AdaMM algorithm proposed in https://arxiv.org/pdf/1904.09237.pdf (1).
    It will be compared to its zero order (ZO) counter-part ZO-AdaMM proposed in https://proceedings.neurips.cc/paper/2019/file/576d026223582a390cd323bef4bad026-Paper.pdf.

    Note that it is an improvement of the well known Adam algorithm.
    Inspired from the source code of the PyTorch's Adam optimizer https://pytorch.org/docs/stable/_modules/torch/optim/adam.html#Adam.
    """

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        if not lr >= 0.0:
            raise ValueError(f'Learning rate must be non-negative, got {lr}')
        if not epsilon >= 0.0:
            raise ValueError(f'Stabilization constant must be non-negative, got {epsilon}')
        if not 0.0 <= beta1 < 1:
            raise ValueError(f'Hyperparameter beta1 must be in [0;1[, got {beta1}')
        if not 0.0 <= beta2 < 1:
            raise ValueError(f'Hyperparameter beta1 must be in [0;1[, got {beta2}')

        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon)
        super(AdaMM, self).__init__(params, defaults)

        # Initialization of exponential moving averages are done lazily

    # Don't need to compute gradients here
    @torch.no_grad()
    def step(self):
        """
        Performs a single optimization step of the AdaMM algorithm.
        """
        # Iterate over the parameters groups in the model
        for group in self.param_groups:
            # Get the parameters for this group (usually only 1 group in our case)
            beta1, beta2 = (group['beta1'], group['beta2'])

            # Iterate over each parameter of the group
            for p in group['params']:
                # If there is gradient for this parameter
                if p.grad is not None:
                    # Get the running quantities for this parameter
                    state = self.state[p]

                    if len(state) == 0:
                        # Lazy initialization to 0 as in (1)
                        # p is a tensor
                        state['gradient_avg'] = torch.zeros_like(p)
                        state['gradient_second_moment'] = torch.zeros_like(p)
                        state['gradient_max_second_moment'] = torch.zeros_like(p)

                        # Keep track of the number of step we updated this parameter
                        state['step'] = 0

                    # Do the updates for this parameter
                    state['gradient_avg'].mul_(beta1).add_(p.grad, alpha=(1 - beta1))
                    state['gradient_second_moment'].mul_(beta2).addcmul_(p.grad, p.grad, value=(1 - beta2))
                    state['step'] += 1

                    # TODO : Correct for the zero-bias ?
                    #grad_avg_hat = state['gradient_avg'] / (1 - beta1 ** state['step'])
                    #grad_second_moment_hat = state['gradient_second_moment'] / (1 - beta2 ** state['step'])

                    state['gradient_max_second_moment_hat'] = torch.maximum(state['gradient_max_second_moment'],
                                                                            state['gradient_second_moment'])

                    # TODO : if F is R^d, then the projection is the identity,
                    #  but then no zero-bias correction as in Adam ?
                    #  See pseudo code from PyTorch
                    p.data.addcdiv_(state['gradient_avg'], state['gradient_second_moment'].sqrt() + group['epsilon'], value=(-group['lr']))
