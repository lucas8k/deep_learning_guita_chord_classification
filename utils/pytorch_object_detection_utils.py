from PIL import Image
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd_predictor
import cv2


def predict_on_image(net, image_path, class_names, threshhold = 0.4):
  net.eval()

  predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)

  orig_image = cv2.imread(image_path)
  image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
  boxes, labels, probs = predictor.predict(image, 5, threshhold)

  labels = [f"{class_names[labels[i]]}: {probs[i]:.2f}" for i in range(boxes.size(0))]
                                                                       
  return draw_boxes_on_image(image, labels, boxes=boxes)

