import train_load
import process_image
import predict
import train
import save
from PIL import Image
import os
import argparse
import torch

parser = argparse.ArgumentParser(description='train a deep learning model')
parser.add_argument('-data_dir','--data_directory', help='Location of data', required=False,default='/home/workspace/ImageClassifier/flowers')
parser.add_argument('-data_dir_image','--data_directory_image', help='Location of test image', required=False,default='/home/workspace/ImageClassifier/flowers/train/1/image_06734.jpg')
parser.add_argument('-check_path','--check_file_path', help='Name of checkpoint file', required=False,default='/home/workspace/ImageClassifier/checkpoint.pth')
parser.add_argument('-epochs','--epoch_number', type=int,help='Number of epches', required=False,default=5)
parser.add_argument('-top_k','--top_classes', type=int,help='Top classes of flowers', required=False,default=5)
parser.add_argument('-save_dir','--save_directory', help='Location to save check point', required=False,default='/home/workspace/ImageClassifier')
parser.add_argument('-arch','--architecture', help='Model architecture', required=False,default='vgg16')
parser.add_argument('-lr','--learning_rate', type=float,help='Description for learning rate', required=False, default=0.01)
parser.add_argument('-device_train','--device_type_train', type=str,help='Description for training device cpu or gpu', required=False, default='gpu')
parser.add_argument('-device_predict','--device_type_predict', type=str,help='Description for inference device cpu or gpu', required=False, default='cpu')
parser.add_argument('-u_units','--hiddenunits', type=int,help='Number of hidden units', required=False,default=4096)
parser.add_argument('-category_names','--categories_names', type=str,help='Name of categories', required=False,default='cat_to_name.json')
args = parser.parse_args()

#The model is trained in this section.
#model = train.train(args.data_directory,args.epoch_number,args.learning_rate,args.device_type_train,args.architecture,args.hiddenunits,args.categories_names)


#Save the model
#save.save(model,args.check_file_path)

#The trained model is loaded.
model = train_load.load_checkpoint(args.check_file_path,args.hiddenunits)
model.eval()

#The sample image is processed.
img_sample = Image.open(args.data_directory_image)
image = process_image.process_image(img_sample)

#The sample image is classified by the model.
predict.predict(model,image,args.top_classes,args.device_type_predict,args.categories_names)