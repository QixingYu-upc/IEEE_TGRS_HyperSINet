import cv2
import numpy as np 
import os
c=0
l=['gcn_branch1','gcn_branch2']
for i in [f1,f2]:
	x_visualize = i.cpu().numpy() #用Numpy处理返回的[1,256,513,513]特征图
	print(l[c],x_visualize.shape)
	x_visualize = np.mean(x_visualize,axis=1).reshape(f1.shape[-2],f1.shape[-1]) #shape为[513,513]，二维
	x_visualize = (((x_visualize - np.min(x_visualize))/(np.max(x_visualize)-np.min(x_visualize)))*255).astype(np.uint8) #归一化并映射到0-255的整数，方便伪彩色化
	x_visualize = cv2.applyColorMap(x_visualize, cv2.COLORMAP_JET)  # 伪彩色处理   
	cv2.imwrite('vis/'+l[c]+'.jpg', x_visualize) #保存可视化图像
	c=c+1
