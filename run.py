import json
import sys
import torch
import torchvision
from torch.nn import CrossEntropyLoss
from optimizers.adamm import AdaMM
from optimizers.zo_adamm import ZO_AdaMM
from torch.utils.data import DataLoader
from utils import train, fix_seeds
sys.path.append('..')

CONFIG_PATH = 'config.json'


def main(use_default_config=True, config=None, zo_optim=False):
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
    fix_seeds(config['seed'])
    ###### REMOVE BEFORE LAST RUNS #######
    #torch.backends.cudnn.deterministic = False
    #torch.backends.cudnn.benchmark = True
    ##########################################
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    if config['verbose']:
        print("Device used: ", device, '\n')

    training_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    validation_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    training_loader = DataLoader(training_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    validation_loader = DataLoader(validation_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2)

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
    else:
        raise ValueError('The chosen net in config is not valid')
    model = model.to(device)
    criterion = CrossEntropyLoss()
    if config['optimizer']=='AdaMM':
        optimizer = torch.optim.Adam(model.parameters(), amsgrad=True)
    elif config['optimizer'] == 'Our-AdaMM':
        optimizer = AdaMM(model.parameters(), lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif config['optimizer'] == 'ZO-AdaMM':
        optimizer = ZO_AdaMM(model.parameters(), lr=config['lr'],
                             betas=(config['beta1'], config['beta2']),
                             mu=config['mu'], eps=1e-8)
    else:
        raise ValueError('The chosen optimizer in config is not valid')

    return train(model, optimizer, criterion, training_loader, validation_loader, device,
                 nb_epochs=config['epochs'], verbose=True, zo_optim=zo_optim)


if __name__ == '__main__':
    config = {
        "seed": 23,
        "batch_size": 128,
        "net": 'b0',
        "optimizer": 'AdaMM',
        "epochs": 100,
        "verbose": True
    }
    main(False, config)