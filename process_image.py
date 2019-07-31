from PIL import Image
import numpy as np

def process_image(image):
    
    width_height_ratio = image.size[0]/image.size[1]
    if image.size[0] > image.size[1]: 
        image.thumbnail((width_height_ratio*256, 256)) 
    else:
        image.thumbnail((256, (1/width_height_ratio)*256))

    margin_left = (image.width-224)/2
    margin_right = margin_left + 224

    margin_bottom = (image.height-224)/2
    margin_top = margin_bottom + 224            

    image = image.crop((margin_left,margin_bottom,margin_right,margin_top))

    image = np.array(image)/255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = (image - mean)/std
    
    image = image.transpose((2, 0, 1))
    
    return image