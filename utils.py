import json
import time

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector


def train(model, optimizer, criterion, training_loader, validation_loader,
          device, nb_epochs, verbose, zo_optim=False, scheduler=None,
          record_weights=False, weights_path=None):
    """
    Train the given model.
    :param model: model to train
    :param optimizer: optimizer to use
    :param criterion: loss function
    :param training_loader: train data loader
    :param validation_loader: validation data loader
    :param device: 'cpu' or 'cuda'
    :param nb_epochs: number of epochs for training
    :param verbose: whether to print progress information
    :param zo_optim: whether ZO optimization is used
    :param scheduler: learning rate scheduler
    :param record_weights: whether to record the weights at each epoch
    :param weights_path: where to save the weights if needed
    :return: train losses, validation losses, validation accuracies, times per epoch
    """
    train_losses = []
    validation_losses = []
    validation_accuracies = []
    epoch_time = []

    if record_weights:
        # Initialize structures to record the weights
        names_sizes = [(name, p.numel()) for name, p in model.named_parameters()]
        weights_sequences = dict()
        for n, s in names_sizes:
            weights_sequences[n] = np.zeros((nb_epochs + 1, s))

        # Record the original weights before training
        for n, p in model.named_parameters():
            weights_sequences[n][0, :] = parameters_to_vector(p).to('cpu').tolist()

    if zo_optim:
        # Global running_loss
        running_loss = 0

    for epoch in range(nb_epochs):
        start = time.time()

        # Training
        model.train()
        training_loss = 0.0

        for data in training_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if zo_optim:
                # Closure used in the ZO optimizer
                running_loss = running_loss + loss.item()
                batch_size = labels.size(0)

                def closure(size_params, mu):
                    grad_est = []

                    # Generate a random direction uniformly on the unit ball or with a gaussian distribution
                    
                    ## ---
                    # The correct way to generate a uniform variable on the sphere is by generating u in this way and then projecting u onto the sphere. 
                    # As we did not immediately find out how to have a uniform variable on the sphere we first used the way that is not commented on.
                    # We found that after doing the experiments how to do it, but after testing it did not change the performance of our algorithm, 
                    # so we left the first version for the sake of reproducibility.
                             
                    #u = torch.normal(mean=torch.zeros(size_params), std=1)  
                    ## ---
                    u = 2 * (torch.rand(size_params) - 0.5)  
                    u.div_(torch.norm(u, "fro"))
                    u = u.to(device)

                    # save the state of the model
                    model_init = dict(model.state_dict())
                    model_init_parameters = model.parameters()

                    # we add to the initial parameters a random perturbation times \mu
                    start_ind = 0
                    for param_tensor in model.parameters():
                        end_ind = start_ind + param_tensor.view(-1).size()[0]
                        param_tensor.add_(u[start_ind:end_ind].view(param_tensor.size()).float(), alpha=mu)
                        start_ind = end_ind

                    # evaluation of the model and the with a random perturbation of the parameters
                    output2 = model(inputs)
                    loss_random = criterion(output2, labels)

                    # compute the "gradient norm"
                    grad_norm = size_params * (loss_random - loss) / mu

                    start_ind = 0
                    for param_tensor in model_init_parameters:
                        end_ind = start_ind + param_tensor.view(-1).size()[0]
                        grad_est.append((grad_norm / batch_size) * u[start_ind:end_ind].view(param_tensor.size()))
                        start_ind = end_ind

                    # reload initial state of the parameters
                    model.load_state_dict(model_init)  # try to subtract the random vector to get back initial params

                    return grad_est

            if not zo_optim:
                loss.backward()
                optimizer.step()
            else:
                optimizer.step(closure)

            training_loss += loss.item()

        train_losses.append(training_loss / len(training_loader))

        # Validation
        model.eval()
        with torch.no_grad():
            correct_preds = 0
            total_preds = 0
            validation_loss = 0
            for inputs, labels in validation_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)

                validation_loss += criterion(outputs, labels).data.item()
                predictions = torch.argmax(outputs, 1)
                total_preds += labels.size(0)
                correct_preds += (predictions == labels).sum().item()

            validation_loss = validation_loss / len(validation_loader)
            validation_losses.append(validation_loss)
            validation_accuracies.append(correct_preds / total_preds)

            if scheduler is not None:
                # Do a learning rate scheduler step
                scheduler.step(validation_loss)

        epoch_time.append(time.time() - start)

        if record_weights:
            # Record the weights of the model
            for n, p in model.named_parameters():
                weights_sequences[n][epoch + 1, :] = parameters_to_vector(p).to('cpu').tolist()

        if verbose and epoch % 5 == 0:
            print(
                f'Epoch: {epoch + 1}/{nb_epochs} |train loss: {train_losses[-1]:.4f} |test loss: {validation_losses[-1]:.4f} |acc: {validation_accuracies[-1]:.4f} |time: {epoch_time[-1]:.4f}')

    if record_weights:
        # Save weights sequence to file
        for n, p in weights_sequences.items():
            np.save(f'{weights_path[:-2]}_{n}_{weights_path[-1]}', p)

    return train_losses, validation_losses, validation_accuracies, epoch_time


def fix_seeds(seed: int):
    """
    Fixes seed for all random functions
    :param seed: int
        Seed to be fixed
    """
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f'Seed set to : {seed}')


def read_json(path):
    """
    Read the given json
    :param path: path of the json file
    :return: json file as a dict
    """
    with open(path, 'r') as f:
        file = json.load(f)

    return file
