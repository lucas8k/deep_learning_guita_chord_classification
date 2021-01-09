from torchvision import datasets, transforms
import torch
import numpy as np

# function for loading and splitting training/test and valid data
def load_data(train_dir, test_dir=None, split=0.25, shuffle=True, batch_size=5, image_size = (224, 224)):
  transfom = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
  


  dataset = datasets.ImageFolder(train_dir,  
                                      transform=transfom)  

  train_set , val_set = torch.utils.data.random_split(dataset, 
                                                      [int(len(dataset) * (1 - split)), 
                                                      len(dataset) - int((len(dataset) * (1 - split)))])

  class_names = dataset.classes
                                             
  dataloaders = {}
  dataset_sizes = {}

  dataloaders['train'] = torch.utils.data.DataLoader(train_set,
                                                    batch_size=batch_size, shuffle=shuffle)
  dataset_sizes['train'] = len(dataloaders['train']) * batch_size

  dataloaders['val'] = torch.utils.data.DataLoader(val_set,
                                                    batch_size=batch_size, shuffle=shuffle)
  
  dataset_sizes['val'] = len(dataloaders['val']) * batch_size

  
  

  if test_dir != None:
    dataset_test = datasets.ImageFolder(test_dir,  
                                      transform=transfom)  
    dataloaders['test'] = torch.utils.data.DataLoader(dataset_test,
                                                    batch_size=batch_size, shuffle=shuffle)
    dataset_sizes['test'] = len(dataloaders['test']) * batch_size

  
  return class_names, dataloaders, dataset_sizes
