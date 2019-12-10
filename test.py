import torch
import torch.nn as nn
import numpy as np
from visdom import Visdom
from datagenerator import dataloader
from torch import optim
from torch.optim.lr_scheduler import StepLR
from model import fpn
from torch.autograd import Variable
import torch.nn.functional as K
import cv2
import os
from utils import encode
batch_size=1
phase='test'
image_size=512

def load_dataset(path,phase='test'):
	dataset=[]
	for root, dirs, files in os.walk(os.path.join(path,phase)):
		print(dirs)
		for file in files:
			if file.endswith(".jpg"):
				print(file)
				dataset.append(os.path.join(root, file))
	return dataset

def transform(image):
	new_image = np.zeros((image_size,image_size,3 ),dtype='uint8')
		
	h,w,c = image.shape
	max_=h
	if max_<w:
		max_=w
	scale = image_size/max_
	if max_==h:
		image = cv2.resize(image,(int(w*scale), int(image_size)))
	else:
		image = cv2.resize(image,( int(image_size),int(h*scale)))
	new_image[0:image.shape[0],0:image.shape[1],:]=image.copy()
	
	new_image = new_image.transpose(2,0,1)
	new_image = torch.from_numpy(new_image).type(torch.FloatTensor)
	new_image = new_image/255;
	new_image = (new_image -0.5)/0.5
	return new_image

	
def read_image(file_name):#,gt_name)
	image = cv2.imread(file_name)
	h,w,c = image.shape
#	gt = cv2.imread(gt_name)
	image = transform(image)
	return image,h,w

def load_model(path=None):
	mdl = fpn()
	if path!=None:
		mdl.load_state_dict(torch.load(path))
	mdl.cuda()
	mdl.eval()
	return mdl

def save_output(output,file_name):
	cv2.imwrite(file_name,output)

	return 0


weight_path ='./weight/0.pt' 
dataset_path='./../dataset'
file_name='./result_original_scale/'
mdl = load_model(weight_path)
dataset = load_dataset(dataset_path)
dg = dataloader(batch_size,dataset_path,phase)
counter=0;
for i in range(1135):
	image,h,w = read_image(dataset[i])
	image = image.unsqueeze(0)
	print("image",image.shape)
	image = image.cuda()
	with torch.no_grad():
		output = mdl(Variable(image))
		output = output.squeeze(0)
		output = output.cpu().numpy()
		output = output.transpose(1,2,0)
		output = cv2.resize(output,(w,h))
		output = output.transpose(2,0,1)
		print(output.shape)
		output = encode(output);
		print(file_name+dataset[i].split("/")[-1].split('.')[-2]+'.png')
		save_output(output,file_name+dataset[i].split("/")[-1].split('.')[-2]+'.png')


