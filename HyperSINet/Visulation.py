# from visualize import visualize_grid_attention_v2
# import numpy as np
#
# # path_model = r"/model1\\"
# ar_load1 = np.load('att.npy')
# Q = np.load('Q.npy')
# Q_cor = Q / np.sum(Q, 0)
# # ar_load = np.mean(ar_load,1).squeeze(0)
# for i in range(ar_load1.shape[1]):
#     ar_load = ar_load1[:, i, :, :].squeeze(0)
#     # attn_mask1 = np.resize(ar_load,(2074,207400))
#     # attn_mask1 = np.matmul(ar_load, Q_cor.T)
#     attn_mask1 = np.matmul(ar_load, Q_cor.T)
#     # print(attn_mask1.shape)
#     # attn_mask = np.mean(attn_mask,0)
#     for j in range(attn_mask1.shape[0]):
#         attn_mask = attn_mask1[j, :]
#         # print(attn_mask.shape)
#         attn_mask = attn_mask.reshape((610, 340))
#         img_path = "D:\\yqx\\Practice\\HyperSINet\\images\\rgb.jpg"
#         save_path = "D:\\yqx\\Practice\\HyperSINet\\images\\"
#         attention_mask = attn_mask
#         visualize_grid_attention_v2(img_path,
#                                     save_path=save_path,
#                                     attention_mask=attention_mask,
#                                     i=i,
#                                     j=j,
#                                     save_image=True,
#                                     save_original_image=True,
#                                     quality=100)

# from visualize import visualize_grid_attention_v2
# import numpy as np
#
# path_model = r"/model1\\"
# ar_load1 = np.load('/home/project/att.npy')
# Q = np.load('/home/project/Q.npy')
# Q_cor = Q / np.sum(Q,0)
# # ar_load = np.mean(ar_load,1).squeeze(0)
# for i in range(2):
#     ar_load = ar_load1[:,i,:,:].squeeze(0)
#     attn_mask1 = np.matmul(ar_load,Q_cor.T)
#     # attn_mask = np.mean(attn_mask,0)
#     for j in range(attn_mask1.shape[0]):
#         attn_mask = attn_mask1[j,:]
#         # print(attn_mask.shape)
#         attn_mask = attn_mask.reshape((610,340))
#         attention_mask = np.zeros((610,340))
#         img_path = "/home/project/rgb.jpg"
#         save_path="/home/project/images/"
#         attention_mask = attn_mask
#         visualize_grid_attention_v2(img_path,
#                                 save_path=save_path,
#                                 attention_mask=attention_mask,
#                                 i=i,
#                                 j=j,
#                                 save_image=True,
#                                 save_original_image=True,
#                                 quality=100)
# import  torch
# print(torch.cuda.is_available())
# import torch
# print(torch.__version__)
#  若返回为True，则使用的是GPU版本的torch，若为False，则为CPU版本

'''Transformer Attention'''

# from visualize import visualize_grid_attention_v2
# import numpy as np
# from sklearn import preprocessing

# path_model = r"/model1\\"
# ar_load1 = np.load('/home/project/att.npy')
# Q = np.load('/home/project/Q.npy')
# Q_cor = Q / np.sum(Q,0)
# # ar_load = np.mean(ar_load,1).squeeze(0)
# for i in range(3,5):
#     ar_load = ar_load1[:,i,:,:].squeeze(0)
#     attn_mask1 = np.matmul(ar_load,Q.T)
#     for j in range(attn_mask1.shape[0]):
#         attn_mask = attn_mask1[j,:]
#         # print(attn_mask.shape)
#         attn_mask = attn_mask.reshape((610,340))
#         img_path = "/home/project/rgb.jpg"
#         save_path="/home/project/images/"
#         attention_mask = attn_mask
#         visualize_grid_attention_v2(img_path,
#                                 save_path=save_path,
#                                 attention_mask=attention_mask,
#                                 i=i,
#                                 j=j,
#                                 save_image=True,
#                                 save_original_image=True,
#                                 quality=100)
# ''' TSNE'''
import numpy as np
import data_reader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

color_map = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
             '#17becf', '#FFEBCD', '#FFF8DC', '#DC143C', '#B8860B', '#8B0000', '#00CED1', '#DCDCDC']  # 7个类，准备7种颜色


def plot_embedding_2D(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    x: dict = {}
    y: dict = {}
    for i in range(np.max(label)):
        x[f"{i + 1}"] = []
        y[f"{i + 1}"] = []
    for i in range(data.shape[0]):
        cc = label[i]
        x[f"{cc}"].append(data[i, 0])
        y[f"{cc}"].append(data[i, 1])
    if np.max(label) == 16:
        A = plt.scatter(x[f"{1}"], y[f"{1}"], color=color_map[1], marker='o', s=0.5)
        B = plt.scatter(x[f"{2}"], y[f"{2}"], color=color_map[2], marker='o', s=0.5)
        C = plt.scatter(x[f"{3}"], y[f"{3}"], color=color_map[3], marker='o', s=0.5)
        D = plt.scatter(x[f"{4}"], y[f"{4}"], color=color_map[4], marker='o', s=0.5)
        E = plt.scatter(x[f"{5}"], y[f"{5}"], color=color_map[5], marker='o', s=0.5)
        F = plt.scatter(x[f"{6}"], y[f"{6}"], color=color_map[6], marker='o', s=0.5)
        G = plt.scatter(x[f"{7}"], y[f"{7}"], color=color_map[7], marker='o', s=0.5)
        H = plt.scatter(x[f"{8}"], y[f"{8}"], color=color_map[8], marker='o', s=0.5)
        I = plt.scatter(x[f"{9}"], y[f"{9}"], color=color_map[9], marker='o', s=0.5)
        J = plt.scatter(x[f"{10}"], y[f"{10}"], color=color_map[10], marker='o', s=0.5)
        K = plt.scatter(x[f"{11}"], y[f"{11}"], color=color_map[11], marker='o', s=0.5)
        L = plt.scatter(x[f"{12}"], y[f"{12}"], color=color_map[12], marker='o', s=0.5)
        M = plt.scatter(x[f"{13}"], y[f"{13}"], color=color_map[13], marker='o', s=0.5)
        N = plt.scatter(x[f"{14}"], y[f"{14}"], color=color_map[14], marker='o', s=0.5)
        O = plt.scatter(x[f"{15}"], y[f"{15}"], color=color_map[15], marker='o', s=0.5)
        P = plt.scatter(x[f"{16}"], y[f"{16}"], color=color_map[16], marker='o', s=0.5)
        plt.xticks([])
        plt.yticks([])
        plt.legend((A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P), (
        'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15', 'c16'),
                   loc="upper right")
        plt.savefig('speformertsne_sa.png')
    elif np.max(label) == 15:
        A = plt.scatter(x[f"{1}"], y[f"{1}"], color=color_map[1], marker='o', s=0.5)
        B = plt.scatter(x[f"{2}"], y[f"{2}"], color=color_map[2], marker='o', s=0.5)
        C = plt.scatter(x[f"{3}"], y[f"{3}"], color=color_map[3], marker='o', s=0.5)
        D = plt.scatter(x[f"{4}"], y[f"{4}"], color=color_map[4], marker='o', s=0.5)
        E = plt.scatter(x[f"{5}"], y[f"{5}"], color=color_map[5], marker='o', s=0.5)
        F = plt.scatter(x[f"{6}"], y[f"{6}"], color=color_map[6], marker='o', s=0.5)
        G = plt.scatter(x[f"{7}"], y[f"{7}"], color=color_map[7], marker='o', s=0.5)
        H = plt.scatter(x[f"{8}"], y[f"{8}"], color=color_map[8], marker='o', s=0.5)
        I = plt.scatter(x[f"{9}"], y[f"{9}"], color=color_map[9], marker='o', s=0.5)
        J = plt.scatter(x[f"{10}"], y[f"{10}"], color=color_map[10], marker='o', s=0.5)
        K = plt.scatter(x[f"{11}"], y[f"{11}"], color=color_map[11], marker='o', s=0.5)
        L = plt.scatter(x[f"{12}"], y[f"{12}"], color=color_map[12], marker='o', s=0.5)
        M = plt.scatter(x[f"{13}"], y[f"{13}"], color=color_map[13], marker='o', s=0.5)
        N = plt.scatter(x[f"{14}"], y[f"{14}"], color=color_map[14], marker='o', s=0.5)
        O = plt.scatter(x[f"{15}"], y[f"{15}"], color=color_map[15], marker='o', s=0.5)
        plt.xticks([])
        plt.yticks([])
        plt.legend((A, B, C, D, E, F, G, H, I, J, K, L, M, N, O),
                   ('c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15'),
                   loc="upper right")
        plt.savefig('speformertsne_hu.png')
    elif np.max(label) == 9:
        A = plt.scatter(x[f"{1}"], y[f"{1}"], color=color_map[1], marker='o', s=0.5)
        B = plt.scatter(x[f"{2}"], y[f"{2}"], color=color_map[2], marker='o', s=0.5)
        C = plt.scatter(x[f"{3}"], y[f"{3}"], color=color_map[3], marker='o', s=0.5)
        D = plt.scatter(x[f"{4}"], y[f"{4}"], color=color_map[4], marker='o', s=0.5)
        E = plt.scatter(x[f"{5}"], y[f"{5}"], color=color_map[5], marker='o', s=0.5)
        F = plt.scatter(x[f"{6}"], y[f"{6}"], color=color_map[6], marker='o', s=0.5)
        G = plt.scatter(x[f"{7}"], y[f"{7}"], color=color_map[7], marker='o', s=0.5)
        H = plt.scatter(x[f"{8}"], y[f"{8}"], color=color_map[8], marker='o', s=0.5)
        I = plt.scatter(x[f"{9}"], y[f"{9}"], color=color_map[9], marker='o', s=0.5)
        plt.xticks([])
        plt.yticks([])
        plt.legend((A, B, C, D, E, F, G, H, I), ('c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'),
                   loc="upper right")
        plt.savefig('GHATtsne_pu.png')


def main(data, label):
    n_samples, n_features = data.shape  # 根据自己的路径合理更改

    print('Begining......')

    # 调用t-SNE对高维的data进行降维，得到的2维的result_2D，shape=(samples,2)
    tsne_2D = TSNE(n_components=2, init='pca', random_state=0)
    result_2D = tsne_2D.fit_transform(data)

    print('Finished......')
    plot_embedding_2D(result_2D, label, 't-SNE')  # 将二维数据用plt绘制出来


if __name__ == "__main__":
    output = np.load('output.npy')
    output = output.reshape(-1, output.shape[2])
    data_gt = data_reader.Houston().truth
    data_gt = data_gt.astype('int').flatten()
    x, y = [], []
    for i in range(data_gt.shape[0]):
        if data_gt[i] != 0:
            x.append(output[i, :])
            y.append(data_gt[i])
    x = np.array(x)
    y = np.array(y)
    main(x, y)


# x: dict = {}
# y: dict = {}
# n = 10
# for i in range(n):
#     x[f"{i}"] = []
#     y[f"{i}"] = []
# for i in range(n):
#     print(x[f"{i}"])
'''
import cv2
import numpy as np
import os

cnnout = np.load('cnnoutput.npy').squeeze(0)

for i in range(cnnout.shape[0]):
    # x_visualize = cnnout[i,:,:]
    x_visualize = np.mean(cnnout,axis=0)
    x_visualize = (((x_visualize - np.min(x_visualize)) / (np.max(x_visualize) - np.min(x_visualize))) * 255).astype(np.uint8)  # 归一化并映射到0-255的整数，方便伪彩色化
    x_visualize = cv2.applyColorMap(x_visualize, cv2.COLORMAP_JET)  # 伪彩色处理
    cv2.imwrite('cnnout/' + str(i) + '.png', x_visualize)  # 保存可视化图像
# c=0
# l=['gcn_branch1','gcn_branch2']
# for i in [f1,f2]:
# 	x_visualize = i.cpu().numpy() #用Numpy处理返回的[1,256,513,513]特征图
# 	print(l[c],x_visualize.shape)
# 	x_visualize = np.mean(x_visualize,axis=1).reshape(f1.shape[-2],f1.shape[-1]) #shape为[513,513]，二维
# 	x_visualize = (((x_visualize - np.min(x_visualize))/(np.max(x_visualize)-np.min(x_visualize)))*255).astype(np.uint8) #归一化并映射到0-255的整数，方便伪彩色化
# 	x_visualize = cv2.applyColorMap(x_visualize, cv2.COLORMAP_JET)  # 伪彩色处理
# 	cv2.imwrite('vis/'+l[c]+'.jpg', x_visualize) #保存可视化图像
# 	c=c+1

'''