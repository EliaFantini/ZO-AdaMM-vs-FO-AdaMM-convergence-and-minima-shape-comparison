class Scheduler:
    """
    Learning rate scheduler. It reduces the learning rate
    by the given factor if the given value (typically the validation loss)
    has not improved for the fixed number of steps.
    If using ZO optimization, it also reduces the mu variable the
    same way it reduces the learning rate.
    """
    def __init__(self, optimizer, mode='min', factor=0.5, patience=5, verbose=False, zo_optim=False):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.zo_optim = zo_optim

        if self.mode == "min":
            # If want to minimize (e.g. loss) the given value
            self.best_value = float('inf')
        elif self.mode == "max":
            # If want to maximize (e.g. accuracy) the given value
            self.best_value = float('-inf')

    def step(self, value):
        """
        Perform a scheduler step.
        :param value: current value of the quantity to track
        """
        if (self.mode == "min" and value <= self.best_value) or (self.mode == "max" and value >= self.best_value):
            # Record the best value seen until now
            self.best_value = value
            self.counter = 0
        else:
            # Did not improved the best value, update the counter
            self.counter += 1
            if self.counter > self.patience:
                # Did not improved the best value for "patience" steps
                # -> reduce the learning rate
                self.counter = 0
                for i, g in enumerate(self.optimizer.param_groups):
                    prev_value = g['lr']
                    g['lr'] = max(self.factor * prev_value, 1e-6)

                    if self.zo_optim:
                        # Reduce the mu variable if using ZO optimization
                        prev_mu = g['mu']
                        g['mu'] = max(self.factor * prev_mu, 1e-6)
                        print(f"Mu reduced from {prev_mu} to {g['mu']} on param_group {i}")

                    if self.verbose:
                        print(f"Learning rate reduced from {prev_value} to {g['lr']} on param_group {i}")
