# custom code to create a valid pascal format out of the dowloaded data
import os
import shutil
import fileinput

def convert_to_pascal_voc(source_dir, dest_dir, classes):

  original_path = source_dir

  if not os.path.exists(dest_dir):
    os.makedirs(os.path.join(dest_dir, "Annotations"))
    os.makedirs(os.path.join(dest_dir, "ImageSets", "Main"))
    os.makedirs(os.path.join(dest_dir, "JPEGImages"))

  with open(os.path.join(dest_dir, "labels.txt"), "x") as file:
    for c in classes:
      file.write(c)
      file.write("\n")
    file.close()

  # move all the files
  source_dir_annotation = [original_path + '/test', original_path + '/train', original_path + '/valid']
  target_dir_annotation = os.path.join(dest_dir, "Annotations")

  for source_dir in source_dir_annotation:  
    file_names = os.listdir(source_dir)
        
    for file_name in file_names:
      if file_name.endswith("xml"):
        shutil.copy(os.path.join(source_dir, file_name), target_dir_annotation)

  source_dir_images = [original_path + '/test', original_path + '/train', original_path + '/valid']
  target_dir_images = os.path.join(dest_dir, "JPEGImages")

  for source_dir in source_dir_images:  
    file_names = os.listdir(source_dir)
        
    for file_name in file_names:
      if file_name.endswith("jpg"):
        shutil.copy(os.path.join(source_dir, file_name), target_dir_images)

  #create image set files
  test_c = 0
  with open(os.path.join(dest_dir, "ImageSets", "Main", "test.txt"), "x") as file:
    for file_name in os.listdir(original_path + "/test"):
      if file_name.endswith("jpg"):
        file.write(file_name[:-4])
        file.write("\n")
        test_c += 1
    file.close()

  train_c = 0
  with open(os.path.join(dest_dir, "ImageSets", "Main", "trainval.txt"), "x") as file:
    for file_name in os.listdir(original_path + "/train"):
      if file_name.endswith("jpg"):
        file.write(file_name[:-4])
        file.write("\n")
        train_c += 1
    file.close()

  val_c = 0
  with open(os.path.join(dest_dir, "ImageSets", "Main", "val.txt"), "x") as file:
    for file_name in os.listdir(original_path +"/valid"):
      if file_name.endswith("jpg"):
        file.write(file_name[:-4])
        file.write("\n")
        val_c += 1
    file.close()

  print("Converted {0} testing images".format(test_c))
  print("Converted {0} training images".format(train_c))
  print("Converted {0} validation images".format(val_c))
