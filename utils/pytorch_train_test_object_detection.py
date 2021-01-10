import time
import torch
import os
import itertools
from functools import cmp_to_key


from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, StepLR

from vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from vision.ssd.ssd import MatchPrior
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.datasets.voc_dataset import VOCDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import mobilenetv1_ssd_config as config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
import gc


# helper function for training 
def train(loader, val_loader, net, criterion, optimizer, scheduler, device, epochs=5, model_prefix="", save_path="", save_model=True, callback=None):
  start = time.localtime()

  best_loss = -1

  for epoch in range(epochs):
      net.train(True)
      print("Training Epoch {} for {} steps:".format(epoch, len(loader)))
      start_epoch = time.localtime()
      scheduler.step()
      running_loss = 0.0
      running_regression_loss = 0.0
      running_classification_loss = 0.0
      for i, data in enumerate(loader):
       
          images, boxes, labels = data
          images = images.to(device)
          boxes = boxes.to(device)
          labels = labels.to(device)

          optimizer.zero_grad()
          confidence, locations = net(images)
          regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  
          loss = regression_loss + classification_loss
          loss.backward()
          optimizer.step()

          running_loss += loss.item()
          running_regression_loss += regression_loss.item()
          running_classification_loss += classification_loss.item()

      avg_loss = running_loss / len(loader)
      avg_reg_loss = running_regression_loss / len(loader)
      avg_clf_loss = running_classification_loss / len(loader)
      print(
          f"Epoch: {epoch}, " +
          f"Avg Train Loss: {avg_loss:.4f}, " +
          f"Avg Train Regression Loss {avg_reg_loss:.4f}, " +
          f"Avg Train Classification Loss: {avg_clf_loss:.4f}"
      )

      running_loss = 0.0
      running_regression_loss = 0.0
      running_classification_loss = 0.0
      

      val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, device)

      print(
          f"Epoch: {epoch}, " +
          f"Validation Loss: {val_loss:.4f}, " +
          f"Validation Regression Loss {val_regression_loss:.4f}, " +
          f"Validation Classification Loss: {val_classification_loss:.4f}"
      )

      if callback != None:
        callback(epoch, (avg_loss, avg_reg_loss, avg_clf_loss), (val_loss, val_regression_loss, val_classification_loss))


      if (best_loss == -1 or val_loss < best_loss) and save_model:
          model_path = os.path.join(save_path, f"{model_prefix}-{epoch}-Loss-{val_loss}.pth")
          net.save(model_path)
          print(f"Saved model {model_path}")
          best_loss = val_loss
      end_epoch = time.localtime()
      print("Training epoch " + str(epoch) + " took " + str(end_epoch.tm_min - start_epoch.tm_min) + " Minutes")

  end = time.localtime()
  print("Training overall took " + str(end.tm_min - start.tm_min) + " Minutes")

#helper function for testing
def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num

def compare_model(model1, model2):
    loss1 = float(model1.split("Loss-",1)[1].replace(".pth", ""))
    loss2 = float(model2.split("Loss-",1)[1].replace(".pth", ""))
    if loss1 < loss2:
        return -1
    elif loss1 > loss2:
        return 1
    else:
        return 0

def get_best_model(dir):
  model_names = [name for name in os.listdir(dir) if name.endswith(".pth")]
  #print(model_names)
  return sorted(model_names, key=cmp_to_key(compare_model))[0]