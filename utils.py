import torch
import time

def train(model, optimizer, criterion, training_loader, validation_loader, device, nb_epochs, verbose):

    train_losses = []
    validation_losses=[]
    validation_accuracies = []
    epoch_time=[]

    for epoch in range(nb_epochs):
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

            loss.backward()
            optimizer.step()

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
            print(f'Epoch: {epoch + 1}/{nb_epochs} |train loss: {train_losses[-1]:.4f} |test loss: {validation_losses[-1]:.4f} |acc: {validation_accuracies[-1]:.4f} |time: {epoch_time[-1]:.4f}')

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