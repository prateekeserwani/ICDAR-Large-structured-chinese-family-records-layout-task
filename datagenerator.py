import numpy as np
import cv2
import os
import torch
from utils import * 

class dataloader():
	def __init__(self,batch_size,path,phase,max_side=2000):
		self.batch_size=batch_size
		self.pointer =0
		self.path = path
		self.dataset = []
		self.phase=phase
		self.load_dataset()
		self.image_size = max_side

	def load_dataset(self):
		for root, dirs, files in os.walk(os.path.join(self.path,self.phase)):
			print(dirs)
			for file in files:
				if file.endswith(".jpg"):
					print(file)
					self.dataset.append((os.path.join(root, file),os.path.join(self.path,'annotation/pixel', file[:-4]+'.png')))

		print(self.dataset)		

	def transform(self,image,gt):
		new_image = np.zeros((self.image_size,self.image_size,3 ),dtype='uint8')
		
		h,w,c = image.shape
		max_=h
		if max_<w:
			max_=w
		scale = self.image_size/max_
		if max_==h:
			image = cv2.resize(image,(int(w*scale), int(self.image_size)))
		else:
			image = cv2.resize(image,( int(self.image_size),int(h*scale)))
		new_image[0:image.shape[0],0:image.shape[1],:]=image.copy()
		
		new_image = new_image.transpose(2,0,1)
		new_image = torch.from_numpy(new_image).type(torch.FloatTensor)
		new_image = new_image/255;
		new_image = (new_image -0.5)/0.5

		gt = decode(gt,self.image_size,scale)
		gt = torch.from_numpy(gt).type(torch.uint8)
		return new_image,gt

	
	def read_image_and_gt(self,file_name,gt_name):
		image = cv2.imread(file_name)
		gt = cv2.imread(gt_name)
		image, gt = self.transform(image,gt)
		return image,gt

	def create_batch(self):
		image_batch = torch.zeros(self.batch_size,3,self.image_size,self.image_size).type(torch.FloatTensor)
		#image_batch_original = torch.zeros(self.batch_size,3,self.image_size,self.image_size).type(torch.FloatTensor)
		gt_batch = torch.zeros(self.batch_size,3,self.image_size,self.image_size).type(torch.uint8)
		batch=0;
		while(1):
			if batch>=self.batch_size:
				break;
			image_name, gt_name = self.dataset[self.pointer]
			self.pointer=(self.pointer+1)%len(self.dataset);
			image, gt = self.read_image_and_gt(image_name, gt_name )
			image_batch[batch,...]=image
			#image_batch_original[batch,...]=image_original
			gt_batch[batch,...]=gt
			batch+=1;
		return image_batch,gt_batch

'''

path='./../dataset/'
batch_size=2
phase='train'
dg = dataloader(batch_size,path,phase)

for i in range(10000):
	image, label = dg.create_batch();
	print('iteration :',i,'image ',image.shape,'label :',label.shape)
'''

