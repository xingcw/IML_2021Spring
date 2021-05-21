import os
import shutil
import torch
import datetime
import numpy as np


def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, backbone,
        metrics=None, start_epoch=0, val_thresh=0.2, save_model=True, save_path=None):

    if save_path is None:
        time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = 'models/' + '_'.join([backbone, time])
        os.makedirs(save_path)
        shutil.copy('networks.py', save_path+'/networks.py')
        shutil.copy('main.py', save_path+'/main.py')

    if metrics is None:
        metrics = []

    best_val = -1

    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics)
        message = f'Epoch: {epoch + 1}/{n_epochs}. Train set: Average loss: {train_loss:.4f}'
        for metric in metrics:
            message += f'\t{metric.name()}: {metric.value()}'

        # Validation stage
        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
        val_loss /= len(val_loader)
        message += f'\nEpoch: {epoch + 1}/{n_epochs}. Validation set: Average loss: {val_loss:.4f}'
        for metric in metrics:
            message += f'\t{metric.name()}: {metric.value()}'
        print(message)
        scheduler.step(val_loss)

        if val_loss <= val_thresh and val_loss < best_val and save_model:
            best_val = val_loss
            model_path = save_path + f'/epoch{epoch}_val_{val_loss}_{backbone}.pth'
            print(f'Saving model to {model_path}')
            state = {
                'epoch': epoch,
                'val_loss': val_loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, model_path)


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, data in enumerate(train_loader):
        target = None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)

        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += f'\t{metric.name()}: {metric.value()}'

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, data in enumerate(val_loader):
            target = None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics
