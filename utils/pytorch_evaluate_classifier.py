import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
import torchvision
import time
import matplotlib.pyplot as plt


# define a function to test the model
def test_model(model, dataloaders, device, dataset_sizes, dataset="val"):
    since = time.time()
    best_acc = 0.0
    model.eval()   # Set model to evaluate mode
    running_corrects = 0

    # Iterate over data.
    for inputs, labels in dataloaders[dataset]:
        inputs = inputs.to(device)
        labels = labels.to(device)


        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        # statistics
        running_corrects += torch.sum(preds == labels.data)

    acc = running_corrects.double() / dataset_sizes[dataset]


    print('Predicting correct {} out of {} images. Acc: {:.4f} on the dataset {}'.format(
        running_corrects, dataset_sizes[dataset], acc, dataset))

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) 

# define a function to visualize predictions
def visualize_model(model, dataloaders, class_names, device, num_images=8, dataset="test"):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders[dataset]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for j in range(inputs.size()[0]):
              images_so_far += 1
              ax = plt.subplot(num_images//2, 2, images_so_far)
              ax.axis('off')
              ax.set_title('predicted: {} --- acutal {}'.format(class_names[preds[j]], class_names[labels[j]]))
              imshow(inputs.cpu().data[j])

              if images_so_far == num_images:
                  model.train(mode=was_training)
                  return
        model.train(mode=was_training)