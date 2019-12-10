# ICDAR-Large-structured-chinese-family-records-layout-task


The used method is based on U-net architecture [1]. The contracting path is derived from a standard ResNet-18 model. From contracting path, we have extracted features from three different depths having feature map’s size F1 (64x128x128), F2 (128x64x64), and F3 (256x32x32). In expanding path, the stacking of convolution (C(x,y) , with x kernels of size y), 
ReLU activation (R), softmax (S), upsampling by factor x (Ux ), and feature map concatenation ([_,Fx ], denotes the feature map from previous layer and feature map Fx ) in expanding path are : C(256,1)RU2 ->[_,F2 ]-> C(256,1)RU2 ->[_,F1 ]-> C(256,1)RU4 ->C(256,1)RC(3,1) ->S. This stacking has been done over F3 feature maps of contracting path. For training, we rescaled the image to 512 X 512 by preserving the aspect ratio of the original image. We have considered the ground truth consists of three classes: text, text-boundary, and background. The text boundary is achieved by subtracting the text pixel image from the dilated text image. The dilated text image is obtained by applying the dilation operation on text pixel image with 5 X 5 kernel and 2 iterations. We have used the dice loss [2], Adam optimizer with an initial learning rate of 10 -3
. The learning rate is decreased by multiplicative factor of 0.1 after every 10k iterations. While prediction, we have taken the threshold of 0.7 for text pixels and remaining pixels are assigned as background.

![Image description](ART.png?raw=true "Title")


# weights 
[https://drive.google.com/file/d/1lWJIsxGjt6L6Vs2uiKTum-eMB4P-QpGA/view?usp=sharing]
