import json
import math
import os
import sys
import torch
import torchvision
from torch.nn import CrossEntropyLoss

from models.scalable_model import ModularModel
from optimizers.adamm import AdaMM
from optimizers.zo_adamm import ZO_AdaMM
from torch.utils.data import DataLoader
from utils import train, fix_seeds, Scheduler
from models.small_model import SmallModel

sys.path.append('..')

CONFIG_PATH = 'config.json'


def main(use_default_config=True, config=None, deterministic=True):
    """
    Main function that loads the data, instantiates data loaders and model, trains the model and
    outputs predictions.
    :param use_default_config: bool
    True to use config.json as config dictionary. Default is True
    :param config: dict
    Dictionary containing all parameters, ignored if use_default_config is set to True. Default is None
    :param zo_optim: boolean
    True to use a zero order optimizer
    """
    if use_default_config:
        config = json.load(open(CONFIG_PATH))
    if config['verbose']:
        print("Running configuration:")
        config_keys = [v for v, m in config.items() if not (v.startswith('_') or callable(m))]
        for key in config_keys:
            print(f"    {key} : {config[key]}")

    if deterministic:
        fix_seeds(config['seed'])

    ###### REMOVE BEFORE LAST RUNS #######
    # torch.backends.cudnn.deterministic = False
    # torch.backends.cudnn.benchmark = True
    ##########################################

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    if config['verbose']:
        print("Device used: ", device, '\n')

    if config['dataset'] == 'mnist':
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        training_dataset = torchvision.datasets.MNIST('data/mnist/', download=True, train=True, transform=transform)
        training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=config['batch_size'])

        validation_dataset = torchvision.datasets.MNIST('data/mnist/', download=True, train=False, transform=transform)
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=config['batch_size'])
    if config['dataset'] == 'cifar':
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

    if config['net'] == 'b0':
        model = torchvision.models.efficientnet_b0()
    elif config['net'] == 'b1':
        model = torchvision.models.efficientnet_b1()
    elif config['net'] == 'b2':
        model = torchvision.models.efficientnet_b2()
    elif config['net'] == 'b3':
        model = torchvision.models.efficientnet_b3()
    elif config['net'] == 'b4':
        model = torchvision.models.efficientnet_b4()
    elif config['net'] == 'b5':
        model = torchvision.models.efficientnet_b5()
    elif config['net'] == 'b6':
        model = torchvision.models.efficientnet_b6()
    elif config['net'] == 'b7':
        model = torchvision.models.efficientnet_b7()
    elif config['net'] == 'small':
        model = SmallModel()
    elif config['net'] == 'scalable':
        model = ModularModel(scale=config['scale'])
    elif config['net'] == 'mobilenet':
        model = torchvision.models.mobilenet_v3_small()
    else:
        raise ValueError('The chosen net in config is not valid')
    model = model.to(device)
    criterion = CrossEntropyLoss()

    if config['optimizer'] == 'AdaMM':
        optimizer = torch.optim.Adam(model.parameters(), amsgrad=True)
    elif config['optimizer'] == 'Our-AdaMM':
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
    if config['use_scheduler']:
        scheduler = Scheduler(optimizer, mode='min', factor=0.5, patience=2, verbose=True, zo_optim=config['zo_optim'])
    else:
        scheduler = None

    d = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"d= {d}, sqrt(d)= {math.sqrt(d)}")

    if config['zo_optim']:
        with torch.no_grad():
            output = train(model, optimizer, criterion, training_loader, validation_loader, device,
                           nb_epochs=config['epochs'], verbose=True, zo_optim=config['zo_optim'], scheduler=scheduler)
    else:
        output = train(model, optimizer, criterion, training_loader, validation_loader, device,
                       nb_epochs=config['epochs'], verbose=True, zo_optim=config['zo_optim'], scheduler=scheduler)
    return output, d


def experiments(config, path, scales, nb_exp=10):
    """
    Run the experiment for all the given scales
    :param config: config for the training
    :param path: where to save the results
    :param scales: what scales to use
    :param nb_exp: number of times we train each model
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
            (train_losses, validation_losses, validation_accuracies, epoch_time), d = main(False, config, deterministic=True)

            res = dict()
            res['train_losses'] = train_losses
            res['validation_losses'] = validation_losses
            res['train_accuracies'] = validation_accuracies
            res['epoch_time'] = epoch_time
            res['nb_params'] = d
            res['seed'] = config['seed']
            tmp.append(res)

            config['seed'] = config['seed'] + 1

        results['values'] = tmp

        # Save the results in file
        with open(os.path.join(path, f'result_{config["optimizer"]}_{s:4f}.json'), 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)


opt_params_zo_adamm = [1e-03, 0.3, 0.5, 1e-12]
opt_params_adamm = [1e-3, 0.9, 0.999, 1e-8]

if __name__ == '__main__':
    config = {
        "dataset": 'mnist',  # cifar, mnist
        "seed": 23,
        "batch_size": 100,
        "net": 'small',
        "optimizer": 'ZO-AdaMM',
        "opt_params": [1e-3, 0.9, 0.999, 1e-8],  # lr,beta1,beta2,epsilon
        "epochs": 100,
        "zo_optim": True,
        "mu": 1e-03,
        "verbose": True,
        "use_scheduler": True
    }
    main(False, config)
"""
d= 2590, sqrt(d)= 50.89204259999789
Our Adam Epoch: epoch 1/12 |train loss: 0.8106 |test loss: 0.2653 |acc: 0.9227 |time: 13.5087
ZO Adam Epoch: epoch 1/100 |train loss: 2.2825 |test loss: 2.2497 |acc: 0.2147 |time: 11.7849
theoretical: train loss: 0.8106* 50.9 = 41.2595, test loss: 0.2653* 50.9 = 13.5037, acc: 0.9227/50.9 = 0.01812
51 epochs:
ZO Adam : epoch 51/100 |train loss: 1.8957 |test loss: 1.8791 |acc: 0.5793 |time: 9.1280


]Epoch: 8/100 |train loss: 1.1512 |test loss: 1.1094 |acc: 0.6592 |time: 35.5637
 1/100 |train loss: 0.1530 |test loss: 0.1277 |acc: 0.9587 |time: 40.6062
"""
