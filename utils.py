import cv2
import numpy as np


def decode(image,max_side,scale):
	# R G B
	# background = 0, 0 , 1
	# text = 0 , 0 , 8
	# dontcare = 128, 0 , 8 


	temp = np.zeros((max_side,max_side),dtype='uint8')		
	plane1 = (image[:,:,2]==0)*1
	plane2 = (image[:,:,1]==0)*1
	plane3 = (image[:,:,0]==8)*1
	#print("plane1",plane1)
	text = plane1*plane2*plane3
	text = text.astype('uint8')

	h,w,c=image.shape
	if h>w:
		text = cv2.resize(text,(int(scale*w),max_side))
	else:	
		text = cv2.resize(text,(max_side,int(scale*h)))
	text = (text>0.6)*1
	text = text.astype('uint8')	

	temp[0:text.shape[0],0:text.shape[1]]=text.copy()

	kernel = np.ones((5,5),np.uint8)
	text_boundary = cv2.dilate(temp,kernel,iterations = 2)

	background =1-text_boundary

	text_boundary = text_boundary - temp
	

	ground_truth = np.stack((temp,text_boundary,background))
	print(ground_truth.shape)
	ground_truth = ground_truth.astype('uint8')
	
	return ground_truth

def encode(output):
	temp=output[0,:,:]
	text_boundary=output[1,:,:]
	background=output[2,:,:]
#	temp1= np.zeros((temp.shape[0],temp.shape[1],3),dtype='uint8')
#	if h>w:
#		temp=cv2.resize(temp,(int(temp.shape[0]/scale),temp.shape[1]))	
#	else:	
#		temp = cv2.resize(temp,(temp.shape[0],int(temp.shape[1]/scale)))
#	temp[0:temp.shape[0],0:text.shape[1]]=text.copy()
	plane_b=((temp>0.7)*8)+((temp<0.7)*1)#+(text_boundary==1)*8
	plane_g=np.zeros((temp.shape[0],temp.shape[1]),dtype='uint8')
	plane_r=np.zeros((temp.shape[0],temp.shape[1]),dtype='uint8')
	print("temp.shape",temp.shape)
	image = np.zeros((temp.shape[0],temp.shape[1],3),dtype='uint8')
	image[:,:,0]=plane_b.copy()
	image[:,:,1]=plane_g.copy()
	image[:,:,2]=plane_r.copy()	
	#print(image.shape)
	#cv2.imwrite("image.jpg",image*20)
	return image

'''	
image = cv2.imread("abc.png")
max_side=1024
h,w,c = image.shape
max_=h
if max_<w:
	max_=w
scale = max_side/max_
cv2.imwrite('image1.jpg',image)
gt = decode(image,max_side,scale)
x=encode(gt)
cv2.imwrite('0.jpg',gt[0,:,:]*255)
cv2.imwrite('1.jpg',gt[1,:,:]*255)
cv2.imwrite('2.jpg',gt[2,:,:]*255)
cv2.imwrite('3.jpg',(gt[1,:,:]*gt[2,...])*200)
'''
