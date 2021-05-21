import os
import shutil
import torch
import datetime


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

    best_val = 1

    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        # Train stage
        returns = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics)
        train_loss, metrics, [easy, semi_hard, hard] = returns
        message = f'Epoch: {epoch + 1}/{n_epochs}  Train set: Average loss: {train_loss:.4f}  '
        for metric in metrics:
            message += f' {metric.name()}: {metric.value():4f}\t'
        message += f'Easy ratio: {easy:.2f}  Semi Hard ratio: {semi_hard:.2f}  Hard ratio: {hard:.2f}'

        # Validation stage
        val_loss, metrics, [easy, semi_hard, hard] = test_epoch(val_loader, model, loss_fn, cuda, metrics)
        val_loss /= len(val_loader)
        message += f'\nEpoch: {epoch + 1}/{n_epochs}  Validation set: Average loss: {val_loss:.4f}\t'
        for metric in metrics:
            message += f'{metric.name()}: {metric.value():4f}\t'
        message += f'Easy ratio: {easy:.2f}  Semi Hard ratio: {semi_hard:.2f}  Hard ratio: {hard:.2f}'
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
    total_loss = 0
    easy_ratio, semi_hard_ratio, hard_ratio = 0, 0, 0
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
        loss, [easy, semi_hard, hard] = loss_outputs
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        easy_ratio += easy / train_loader.batch_size
        semi_hard_ratio += semi_hard / train_loader.batch_size
        hard_ratio += hard / train_loader.batch_size

        for metric in metrics:
            metric(outputs, target, loss.item())

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                batch_idx * train_loader.batch_size * log_interval, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item())
            for metric in metrics:
                message += f' {metric.name()}: {metric.value():.6f}\t'
            batch_size = train_loader.batch_size
            message += f'Easy ratio: {easy/batch_size:.2f}  Semi Hard ratio: {semi_hard/batch_size:.2f}  ' \
                       f'Hard ratio: {hard/batch_size:.2f}'
            print(message)

    total_loss /= len(train_loader)
    easy_ratio /= len(train_loader)
    semi_hard_ratio /= len(train_loader)
    hard_ratio /= len(train_loader)

    return total_loss, metrics, [easy_ratio, semi_hard_ratio, hard_ratio]


def test_epoch(val_loader, model, loss_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        easy_ratio, semi_hard_ratio, hard_ratio = 0, 0, 0
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
            loss, [easy, semi_hard, hard] = loss_outputs
            val_loss += loss.item()
            easy_ratio += easy / val_loader.batch_size
            semi_hard_ratio += semi_hard / val_loader.batch_size
            hard_ratio += hard / val_loader.batch_size

            for metric in metrics:
                metric(outputs, target, loss.item())
                
        easy_ratio /= len(val_loader)
        semi_hard_ratio /= len(val_loader)
        hard_ratio /= len(val_loader)
        
    return val_loss, metrics, [easy_ratio, semi_hard_ratio, hard_ratio]
