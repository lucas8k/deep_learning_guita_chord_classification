import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
import os
import torchvision
import time


# define a function to train the model
def train_model(model, device, criterion, optimizer, dataloaders, dataset_sizes, scheduler=None, num_epochs=5, model_prefix="MODEL", save_path=None, callback=None):
    since = time.time()

    best_acc = 0.0
    best_loss = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        train_acc = -1
        train_loss = -1
        val_acc = -1
        val_loss = -1

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train' and scheduler != None:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'val':
                val_acc = epoch_acc.item()
                val_loss = epoch_loss
            elif phase == 'train':
                train_acc = epoch_acc.item()
                train_loss = epoch_loss
            if phase == 'val' and best_loss > val_loss:
                best_loss = val_loss
                if save_path != None:
                  model_path = os.path.join(save_path, f"{model_prefix}-{epoch}-Loss-{epoch_loss}.pth")
                  torch.save(model.state_dict(), model_path)
                  print(f"Saved model {model_path}")
              
              
        if callback != None:
          callback((train_loss, train_acc), (val_loss, val_acc))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return model


