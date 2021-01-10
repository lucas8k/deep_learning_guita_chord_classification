import torch
import os
import itertools

from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, StepLR

from vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from vision.ssd.ssd import MatchPrior
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.datasets.voc_dataset import VOCDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import mobilenetv1_ssd_config as config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform

def load_dataset(dataset_path, labels_path, batch_size=30, shuffle=True):
  # create data transforms for train/test/val
  train_transform = TrainAugmentation(
      config.image_size, config.image_mean, config.image_std)
  target_transform = MatchPrior(config.priors, config.center_variance,
                                config.size_variance, 0.5)

  test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

  datasets = []
  dataset = VOCDataset(dataset_path, transform=train_transform,
                        target_transform=target_transform)
  store_labels(labels_path, dataset.class_names)
  num_classes = len(dataset.class_names)

  datasets.append(dataset)

  # create training dataset
  train_dataset = ConcatDataset(datasets)
  train_loader = DataLoader(train_dataset, batch_size,
                            num_workers=1,
                            shuffle=shuffle)

  # create validation dataset                           
  val_dataset = VOCDataset(dataset_path, transform=test_transform,
                            target_transform=target_transform, is_test=True)

  val_loader = DataLoader(val_dataset, batch_size,
                          num_workers=1,
                          shuffle=shuffle)
  return train_loader, val_loader, num_classes