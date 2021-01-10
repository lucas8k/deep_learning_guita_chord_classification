#defining a few helper functions to visualize the data
from PIL import Image
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imgaug.augmenters as iaa
import os
import random
import string

def load_boxes(filename):
  root = ET.parse(filename + ".xml").getroot()
  bboxes = []
  labels = []
  for object_tag in root.findall('object'):
    bbox_tag = object_tag.find('bndbox')
    labels.append(object_tag.find('name').text)
    bboxes.append((
        int(bbox_tag.find("xmin").text),
        int(bbox_tag.find("ymin").text),
        int(bbox_tag.find("xmax").text),
        int(bbox_tag.find("ymax").text)
      )
    )
  return bboxes, labels

def draw_boxes_on_image(image, labels=[], boxes=[]):
    for i in range(len(boxes)):
      cv2.rectangle(image, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), (255, 255, 0), 4)
      if labels != None:
        cv2.putText(image, labels[i],
                      (int(boxes[i][0]) + 20, int(boxes[i][1]) + 40),
                      cv2.FONT_HERSHEY_SIMPLEX,
                      1, 
                      (255, 0, 255),
                      2)  
    return image

def load_image(filename, draw_boxes=False, np_array=False):
  orig_image = np.array(Image.open(filename + ".jpg"))
  bboxes, labels = load_boxes(filename)
  if draw_boxes:
    draw_boxes_on_image(orig_image, labels=labels, boxes=bboxes)
  image = orig_image
  if not np_array:
    image = Image.fromarray(orig_image)
  return bboxes, labels, image


def save_bounding_box(new_boxes, org_files, dest_file):
  tree = ET.parse(org_files)
  root = tree.getroot()
  i = 0
  for object_tag in root.iter('object'):
    bbox_tag = object_tag.find('bndbox')
    bbox_tag.find("xmin").text = (str(new_boxes[i]["xmin"]))
    bbox_tag.find("xmax").text = (str(new_boxes[i]["xmax"]))
    bbox_tag.find("ymin").text = (str(new_boxes[i]["ymin"]))
    bbox_tag.find("ymax").text = (str(new_boxes[i]["ymax"]))
    i += 1

  tree.write(dest_file)

def apply_augumentation(filename, seq, multiply=1, save=False, source_dir="", dest_dir=""):
  aug_images = []
  for c in range(multiply):

    boxes, labels, image = load_image(os.path.join(source_dir, filename), draw_boxes=False, np_array=True)
    annotation_name = filename + ".xml"

    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=bbox[0], x2=bbox[2], y1=bbox[1], y2=bbox[3]) for bbox in boxes
    ], shape=image.shape)

    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
    aug_images.append((image_aug, bbs_aug.to_xyxy_array(), labels))

    if save:
      prefix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
      image = Image.fromarray(image_aug)
      image.save(os.path.join(dest_dir, "aug-" + prefix + filename + ".jpg"))

      save_bounding_box([{"xmin": int(bbox[0]),
              "xmax": int(bbox[2]),
              "ymin": int(bbox[1]),
              "ymax": int(bbox[3])} for bbox in bbs_aug.to_xyxy_array()], 
              os.path.join(source_dir, annotation_name),
              os.path.join(dest_dir, "aug-" + prefix  + annotation_name))
  return aug_images