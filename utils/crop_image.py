from PIL import Image 

# crops image by bounding box and adds pading
def crop_image(org_frame, box, padding_x_axis_perc = 3, padding_y_axis_perc = 5):

    image_size_x = org_frame.shape[1]
    image_size_y = org_frame.shape[0]

    padding_x_axis = image_size_x * (padding_x_axis_perc / 100)
    padding_y_axis = image_size_y * (padding_y_axis_perc / 100) 
       
    point1 = [box[0], box[1]]
    point2 = [box[2], box[3]]
    
    if box[0] - padding_x_axis > 0:
        point1[0] = box[0] - padding_x_axis
    else:
        point1[0] = 0 
    
    if box[1] - padding_y_axis > 0:
        point1[1] = box[1] - padding_y_axis
    else:
        point1[1] = 0 
        

    if box[2] + padding_x_axis < image_size_x:
        point2[0] = box[2] + padding_x_axis
    else:
        point2[0] = image_size_x
    
    if box[3] + padding_y_axis < image_size_y:
        point2[1] = box[3] + padding_y_axis
    else:
        point2[1] = image_size_y
        
    return Image.fromarray(org_frame).crop((point1[0], point1[1], point2[0], point2[1]))




