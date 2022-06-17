import json
import math
import os
import sys

import torch
import torchvision
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from models.scalable_model import ModularModel
from models.small_model import SmallModel
from optimizers.adamm import AdaMM
from optimizers.zo_adamm import ZO_AdaMM
from utils import train, fix_seeds
from optimizers.scheduler import Scheduler

sys.path.append('..')

CONFIG_PATH = 'config.json'


def main(use_default_config=True, config=None, deterministic=True, record_weights=False, weights_path=None, init=False):
    """
    Main function that loads the data, instantiates data loaders and model, trains the model and
    outputs predictions.
    :param use_default_config: bool
    True to use config.json as config dictionary. Default is True
    :param config: dict
    Dictionary containing all parameters, ignored if use_default_config is set to True. Default is None
    :param deterministic: boolean
    True to set the seed for the random methods. Default is True
    :param record_weights: boolean
    True to record the weights every 5 epochs
    :param weights_path: string
    Where to save the weights if needed
    :param init: boolean
    Whether to initialize the model weights
    """
    if use_default_config:
        config = json.load(open(CONFIG_PATH))
    if config['verbose']:
        print("Running configuration:")
        config_keys = [v for v, m in config.items() if not (v.startswith('_') or callable(m))]
        for key in config_keys:
            print(f"    {key} : {config[key]}")

    if deterministic:
        # Set the different parameters
        # to make the methods deterministic
        fix_seeds(config['seed'])

    # Use GPU if available
    device = ('cuda' if torch.cuda.is_available() and not config['zo_optim'] else 'cpu')
    #device = ('cuda' if torch.cuda.is_available() else 'cpu')
    if config['verbose']:
        print("Device used: ", device, '\n')

    # Prepare the dataset
    if config['dataset'] == 'mnist':
        # Prepare the MNIST dataset
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        training_dataset = torchvision.datasets.MNIST('data/mnist/', download=True, train=True, transform=transform)
        training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=config['batch_size'])

        validation_dataset = torchvision.datasets.MNIST('data/mnist/', download=True, train=False, transform=transform)
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=config['batch_size'])
    if config['dataset'] == 'cifar':
        # Prepare the CIFAR-10 dataset
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        training_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        validation_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                                          transform=transform)
        training_loader = DataLoader(training_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
        validation_loader = DataLoader(validation_dataset, batch_size=config['batch_size'], shuffle=False,
                                       num_workers=2)

    # Prepare the model
    elif config['net'] == 'small':
        model = SmallModel()
    elif config['net'] == 'scalable':
        model = ModularModel(scale=config['scale'], init=init)
    else:
        raise ValueError('The chosen net in config is not valid')
    model = model.to(device)
    criterion = CrossEntropyLoss()

    # Prepare the optimizer
    if config['optimizer'] == 'AdaMM':
        # PyTorch implementation of AdaMM
        optimizer = torch.optim.Adam(model.parameters(), amsgrad=True)
    elif config['optimizer'] == 'Our-AdaMM':
        # Our implementation of AdaMM
        with torch.no_grad():
            optimizer = AdaMM(model.parameters(), lr=config['opt_params'][0],
                              beta1=config['opt_params'][1], beta2=config['opt_params'][2],
                              epsilon=config['opt_params'][3])
    elif config['optimizer'] == 'ZO-AdaMM':
        optimizer = ZO_AdaMM(model.parameters(), lr=config['opt_params'][0],
                             betas=(config['opt_params'][1], config['opt_params'][2]),
                             eps=config['opt_params'][3],
                             mu=config['mu'])
    else:
        raise ValueError('The chosen optimizer in config is not valid')

    # Set up the learning rate scheduler
    if config['use_scheduler']:
        scheduler = Scheduler(optimizer, mode='min', factor=0.5, patience=2, verbose=True, zo_optim=config['zo_optim'])
    else:
        scheduler = None

    d = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"d= {d}, sqrt(d)= {math.sqrt(d)}")

    if config['zo_optim']:
        # Disable the gradient computations for ZO optimization
        with torch.no_grad():
            output = train(model, optimizer, criterion, training_loader, validation_loader, device,
                           nb_epochs=config['epochs'], verbose=True, zo_optim=config['zo_optim'], scheduler=scheduler,
                           record_weights=record_weights, weights_path=weights_path)
    else:
        output = train(model, optimizer, criterion, training_loader, validation_loader, device,
                       nb_epochs=config['epochs'], verbose=True, zo_optim=config['zo_optim'], scheduler=scheduler,
                       record_weights=record_weights, weights_path=weights_path)
    return output, d


def experiments(config, path, scales, nb_exp=10, record_weights=False, weights_path=None):
    """
    Run the experiments for the given scales.
    :param config: config for the training
    :param path: where to save the results
    :param scales: what scales to use
    :param nb_exp: number of times we train each model
    :param record_weights: whether to record the weights every 5 epochs
    :param weights_path: where to save the weights if needed
    """
    seed_init = config['seed']

    for s in scales:
        # Set the scale of the model
        config['scale'] = s

        print(f'Scale set to : {s}')

        # Save the results
        results = dict()
        results['config'] = config
        tmp = []

        config['seed'] = seed_init

        for i in range(nb_exp):
            # Train the model
            (train_losses, validation_losses, validation_accuracies, epoch_time), d = main(False, config,
                                                                                           deterministic=True,
                                                                                           record_weights=record_weights,
                                                                                           weights_path=f'{weights_path}_{s}_{i}',
                                                                                           init=True)

            # Record the results
            res = dict()
            res['train_losses'] = train_losses
            res['validation_losses'] = validation_losses
            res['train_accuracies'] = validation_accuracies
            res['epoch_time'] = epoch_time
            res['nb_params'] = d
            res['seed'] = config['seed']
            tmp.append(res)

            # Change the seed
            config['seed'] = config['seed'] + 1

        results['values'] = tmp

        # Save the results in file
        with open(os.path.join(path, f'result_{config["optimizer"]}_{s:4f}.json'), 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
