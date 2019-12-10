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
path='./../dataset/'
batch_size=1
phase='train'


def convert(image):
	image = ((image+1)/2)*255
	image = image.type(torch.uint8)
	return image
'''
def loss_function1(y_true, y_pred, smooth=1e-7):
	print(y_true.shape)
	y_true_f = y_true.view(-1)
	y_pred_f = y_pred.view(-1)
	intersect = torch.sum(y_true_f * y_pred_f, axis=-1)
	denom = torch.sum(y_true_f + y_pred_f, axis=-1)
	return torch.mean((2. * intersect / (denom + smooth)))
'''

def loss_function(probas,true_1_hot,eps=1e-7):
	dims = (0,) + tuple(range(2, 4))
	intersection=torch.sum(probas*true_1_hot,0)
	intersection=torch.sum(intersection,-1)
	intersection=torch.sum(intersection,-1)
	cardinality = torch.sum(probas + true_1_hot,0)
	cardinality = torch.sum(cardinality,-1)
	cardinality = torch.sum(cardinality,-1)
	dice_loss = (2. * intersection / (cardinality + eps)).mean()
	loss = (1 - dice_loss)
	return loss

def load_model(path=None):
	mdl = fpn()
##	mdl.cuda()
	mdl.train()
	if path!=None:
		pass;
	return mdl

mdl = load_model()
dg = dataloader(batch_size,path,phase)
iteartion_per_epoch = len(dg.dataset)//batch_size;
optimizer = optim.Adam(mdl.parameters(), lr=0.001)
schedular = StepLR(optimizer, step_size=10000, gamma=0.1)
viz = Visdom()
counter=0;
for i in range(iteartion_per_epoch*10):
	image, label = dg.create_batch();
#	image = image.cuda()
#	label = label.cuda()
	label = label.type(torch.FloatTensor).cuda()
	optimizer.zero_grad()
	output = mdl(Variable(image))
	loss = torch.mean(loss_function(output[:,0,:,:],label[:,0,:,:])+loss_function(output[:,1,:,:],label[:,1,:,:])+loss_function(output[:,2,:,:],label[:,2,:,:]))
	print("loss",loss)
	loss.backward()
	optimizer.step()
	schedular.step()
	print('iteration :',i,'loss ',loss.item())
	if i%100==0 and i>0:
		viz.image(convert(image[0,...]),"image")
		viz.heatmap(convert(output[0,0,:,:]),"heatmap1")
		viz.heatmap(convert(output[0,1,:,:]),"heatmap2")
		viz.heatmap(convert(output[0,2,:,:]),"heatmap3")
	if i%5000==0 and i>0:
		torch.save(mdl.state_dict(), './weight/'+str(counter)+'.pt')
	counter+=1;


