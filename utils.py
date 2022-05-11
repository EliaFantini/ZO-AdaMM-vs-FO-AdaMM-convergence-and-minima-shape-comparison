import numpy as np
import torch
import time


def train(model, optimizer, criterion, training_loader, validation_loader,
          device, nb_epochs, verbose, zo_optim=False):
    train_losses = []
    validation_losses = []
    validation_accuracies = []
    epoch_time = []

    if zo_optim:
        # global running_loss
        running_loss = 0
        lr_init = optimizer.param_groups[0]['lr']
        mu_init = optimizer.param_groups[0]['mu']

    for epoch in range(nb_epochs):

        if zo_optim:
            # Change the learning rate, TODO : use scheduler
            optimizer.param_groups[0]['lr'] = max(lr_init / (np.sqrt(10) ** (epoch / 4.)), 1e-5)
            optimizer.param_groups[0]['mu'] = max(mu_init / (np.sqrt(10) ** (epoch / 4.)), 1e-5)

        start = time.time()

        # Training
        model.train()
        training_loss = 0.0

        for data in training_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if zo_optim:
                running_loss = running_loss + loss.item()
                batch_size = labels.size(0)

                def closure(size_params, mu):
                    grad_est = []

                    # Generate a random direction uniformly on the unit ball or with a gaussian distribution
                    # u = torch.normal(mean = torch.zeros(size_params),std = 100)
                    u = 2 * (torch.rand(size_params) - 0.5)  # need small modif in order to be on the unit sphere
                    u.div_(torch.norm(u, "fro"))

                    # save the state of the model
                    model_init = dict(model.state_dict())
                    model_init_parameters = model.parameters()
                    grad_norm = 0

                    # we add to the initial parameters a random perturbation times \mu
                    start_ind = 0
                    for param_tensor in model.parameters():
                        end_ind = start_ind + param_tensor.view(-1).size()[0]
                        param_tensor.add_(u[start_ind:end_ind].view(param_tensor.size()).float(), alpha=mu)
                        start_ind = end_ind

                    # evaluation of the model and the with a random perturbation of the parameters
                    output2 = model(input)
                    loss_random = criterion(output2, labels)

                    # compute the "gradient norm"
                    # when u is uniform random variable
                    grad_norm = size_params * (loss_random - loss) / mu

                    # when u is Gaussian random variable
                    # grad_norm += (loss_random-loss_init)/mu

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

        epoch_time.append(time.time() - start)
        if verbose:
            print(
                f'Epoch: {epoch + 1}/{nb_epochs} |train loss: {train_losses[-1]:.4f} |test loss: {validation_losses[-1]:.4f} |acc: {validation_accuracies[-1]:.4f} |time: {epoch_time[-1]:.4f}')

    return train_losses, validation_losses, validation_accuracies, epoch_time


def fix_seeds(seed: int):
    """
    Fixes seed for all random functions
    @param seed: int
        Seed to be fixed
    """
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
