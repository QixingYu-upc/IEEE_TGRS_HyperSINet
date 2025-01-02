import numpy as np
import scipy.io as sio
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from skimage.segmentation import slic, mark_boundaries
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv
import time
import warnings
import random
import data_reader
import HyperSINet
import utils
import matplotlib.pyplot as plt
from GNN import split_data_graph as split_data
from GNN import create_graph
from GNN import dr_slic
from torchsummaryX import summary
from thop import profile
# from visualize import visualize_grid_attention_v2
warnings.filterwarnings("ignore")


def seed_torch(seed=128,deter=False):
    '''
    `deter` means use deterministic algorithms for GPU training reproducibility,
    if set `deter=True`, please set the environment variable `CUBLAS_WORKSPACE_CONFIG` in advance
    '''
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(deter)

# seed_torch()
def load_data():
    data = data_reader.Houston().normal_cube
    data_gt = data_reader.Houston().truth
    data_gt = data_gt.astype('int')
    return data,data_gt

data, data_gt = load_data()
class_num = np.max(data_gt)
gt_reshape = np.reshape(data_gt, [-1])
samples_type = ['ratio','same_num'][0]
train_ratio = 0.05
val_ratio = 0.05
train_num = 10
val_num = class_num
learning_rate = 0.001
max_epoch = 1000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path_model = r"/model1"
path_data = None
height,width,bands = data.shape

def Get_mat():
    # split data
    train_index, val_index, test_index = split_data.split_data(gt_reshape, class_num, train_ratio, val_ratio, train_num,val_num, samples_type)  # 划分训练集、验证集和测试集,得到属于三种集的坐标
    train_samples_gt, test_samples_gt, val_samples_gt = create_graph.get_label(gt_reshape, train_index, val_index,test_index)  # ground_truth划分为训练集、验证机和测试集，维度是21025，根据坐标得到属于各种集的类别
    train_gt = np.reshape(train_samples_gt, [height, width])
    test_gt = np.reshape(test_samples_gt, [height, width])
    val_gt = np.reshape(val_samples_gt,[height, width])  # 维度都是[145,145],reshape之后，对应的训练集，测试集，验证集都显现出来了。训练集上，测试集和验证集对应位置为0

    train_samples_gt_onehot = create_graph.label_to_one_hot(train_gt, class_num)
    test_samples_gt_onehot = create_graph.label_to_one_hot(test_gt, class_num)
    val_samples_gt_onehot = create_graph.label_to_one_hot(val_gt,class_num)  # 维度都是[145,145,16]，假如(x,y)属于第9类，那么[x,y,9]即为1

    train_samples_gt_onehot = np.reshape(train_samples_gt_onehot, [-1, class_num]).astype(int)
    test_samples_gt_onehot = np.reshape(test_samples_gt_onehot, [-1, class_num]).astype(int)
    val_samples_gt_onehot = np.reshape(val_samples_gt_onehot, [-1, class_num]).astype(int)  # 维度都是[21025,16]
    train_label_mask, test_label_mask, val_label_mask = create_graph.get_label_mask(train_samples_gt, test_samples_gt,val_samples_gt, data_gt,class_num)  # [21025,16]

    train_samples_gt = torch.from_numpy(train_samples_gt.astype(np.float32)).to(device)
    test_samples_gt = torch.from_numpy(test_samples_gt.astype(np.float32)).to(device)
    val_samples_gt = torch.from_numpy(val_samples_gt.astype(np.float32)).to(device)

    train_samples_gt_onehot = torch.from_numpy(train_samples_gt_onehot.astype(np.float32)).to(device)
    test_samples_gt_onehot = torch.from_numpy(test_samples_gt_onehot.astype(np.float32)).to(device)
    val_samples_gt_onehot = torch.from_numpy(val_samples_gt_onehot.astype(np.float32)).to(device)

    train_label_mask = torch.from_numpy(train_label_mask.astype(np.float32)).to(device)
    test_label_mask = torch.from_numpy(test_label_mask.astype(np.float32)).to(device)
    val_label_mask = torch.from_numpy(val_label_mask.astype(np.float32)).to(device)

    net_input = np.array(data, np.float32)
    net_input = torch.from_numpy(net_input.astype(np.float32)).to(device)
    return train_samples_gt,test_samples_gt,val_samples_gt,train_samples_gt_onehot,test_samples_gt_onehot,val_samples_gt_onehot,train_label_mask,test_label_mask,val_label_mask,net_input

def get_Q_and_S_and_Segments(data, scale=50, compactness=1, max_iter=20, sigma=1, min_size_factor=0.1,max_size_factor=2):
    height, width, bands = data.shape
    n_segments = height * width / scale  # 分割数
    data = np.reshape(data, [height * width, bands])
    minMax = preprocessing.StandardScaler()
    data = minMax.fit_transform(data)
    data = np.reshape(data, [height, width, bands])
    img = data
    segments = slic(img, n_segments=n_segments, compactness=compactness, max_iter=max_iter,
                        convert2lab=False, sigma=sigma, enforce_connectivity=True,
                        min_size_factor=min_size_factor, max_size_factor=max_size_factor, slic_zero=False,
                        start_label=0)
    superpixel_count = segments.max() + 1
    # print("superpixel_count", superpixel_count)
    #####################################显示超像素图片######################################
    # out = mark_boundaries(img[:,:,[57,34,3]], segments)
    # plt.figure()
    # plt.imshow(out)
    # # plt.show()
    # plt.savefig('seg'+str(superpixel_count)+'.png',dpi=300)
    #####################################显示超像素图片######################################
    mask = np.zeros([superpixel_count, superpixel_count])
    for i in range(height - 2):
        for j in range(width - 2):
            sub = segments[i:i + 2, j:j + 2]
            sub_max = np.max(sub).astype(np.int32)
            sub_min = np.min(sub).astype(np.int32)
            if sub_max != sub_min:
                mask[sub_max, sub_min] = mask[sub_min, sub_max] = 1
    # mask[np.where(mask == 0)] = -100
    # mask[np.where(mask == 1)] = 0
    init_seg = segments
    segments = np.reshape(segments, [-1])
    S = np.zeros([superpixel_count, bands], dtype=np.float32)
    Q = np.zeros([width * height, superpixel_count], dtype=np.float32)
    x = np.reshape(img, [-1, bands])
    for i in range(superpixel_count):
        idx = np.where(segments == i)[0]
        count = len(idx)
        pixels = x[idx]
        superpixel = np.sum(pixels, 0) / count  # 同一个块的像素均值做为中心像素的特征值
        S[i] = superpixel
        Q[idx, i] = 1
    A = np.zeros([superpixel_count, superpixel_count], dtype=np.float32)
    (h, w) = init_seg.shape  # [145,145]
    for i in range(h - 2):
        for j in range(w - 2):
            sub = init_seg[i:i + 2, j:j + 2]  # 周围八邻域
            sub_max = np.max(sub).astype(np.int32)
            sub_min = np.min(sub).astype(np.int32)
            if sub_max != sub_min:
                idx1 = sub_max
                idx2 = sub_min
                if A[idx1, idx2] != 0:
                    continue
                pix1 = S[idx1]
                pix2 = S[idx2]
                diss = np.exp(-np.sum(np.square(pix1 - pix2)) / sigma ** 2)
                A[idx1, idx2] = A[idx2, idx1] = diss
    # np.save('Q.npy',Q)
    # row, col = np.diag_indices_from(A)
    # A[row,col] = 1
    # for i in range(superpixel_count):
    #     A[i,:] = (A[i,:] - np.mean(A[i,:])) / np.std(A[i,:])  # 第0列均值方差归一化
    Q = torch.from_numpy(Q.astype(np.float32)).to(device)
    A = torch.from_numpy(A.astype(np.float32)).to(device)
    mask = torch.from_numpy(mask.astype(np.float32)).to(device)
    return Q, A
for ooo in range(8):
    train_samples_gt,test_samples_gt,val_samples_gt,train_samples_gt_onehot,test_samples_gt_onehot,val_samples_gt_onehot,train_label_mask,test_label_mask,val_label_mask,net_input = Get_mat()
    Q,mask = get_Q_and_S_and_Segments(data)
    net = HyperSINet.HyperSINet(Q, mask, in_channels=bands,out_channels=64,layers=2,num_classes=class_num,depth=1).to(device)
    # flops, params = profile(net, (net_input,))
    # print('flops: ', flops, 'params: ', params)
    # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0/net_input.shape[0], params / 1000000.0))
    # print('flops: ', flops, 'params: ', params)
    optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate,weight_decay=1e-3)
    zeros = torch.zeros([height*width]).to(device).float()
    best_loss = 999999
    net.train()
    tic1 = time.time()
    print('go go go')
    for i in range(max_epoch + 1):
        optimizer.zero_grad()
        output = net(net_input)
        loss = utils.compute_loss(output,train_samples_gt_onehot,train_label_mask)
        loss.backward(retain_graph=False)
        optimizer.step()
        with torch.no_grad():
            net.eval()
            output = net(net_input)
            trainloss = utils.compute_loss(output,train_samples_gt_onehot,train_label_mask)
            trainOA = utils.evaluate_performance(output,train_samples_gt,train_samples_gt_onehot,zeros)
            valloss = utils.compute_loss(output, val_samples_gt_onehot, val_label_mask)
            valOA = utils.evaluate_performance(output, val_samples_gt, val_samples_gt_onehot, zeros)
            if valloss < best_loss:
                best_loss = valloss
                torch.save(net.state_dict(),path_model + r"model.pt")
        torch.cuda.empty_cache()
        net.train()
        # if i%200==0:
        #     print("{}\ttrain loss={:.4f}\t train OA={:.4f} val loss={:.4f}\t val OA={:.4f}".format(str(i + 1), trainloss,trainOA, valloss, valOA))
        toc1 = time.time()
    print("\n\n====================training done. starting evaluation...========================\n")
    torch.cuda.empty_cache()
    with torch.no_grad():
        zeros = torch.zeros([height * width]).to(device).float()
        net.load_state_dict(torch.load(path_model + r"model.pt"))
        net.eval()
        tic2 = time.time()
        output = net(net_input)
        output = torch.squeeze(output)
        toc2 = time.time()
        testloss = utils.compute_loss(output,test_samples_gt_onehot,test_label_mask)
        testOA = utils.evaluate_performance(output,test_samples_gt,test_samples_gt_onehot,zeros)
        print("{}\ttest loss={:.4f}\t test OA={:.4f}".format(str(i + 1), testloss, testOA))
    torch.cuda.empty_cache()
    train_time = toc1 - tic1
    test_time = toc2 - tic2

    print("Train time :%d"%train_time)
    print("Test time :%d"%test_time)

    test_label_mask_cpu = test_label_mask.cpu().numpy()[:, 0].astype('bool')
    test_samples_gt_cpu = test_samples_gt.cpu().numpy().astype('int64').T
    predict = torch.argmax(output, dim=1).cpu().numpy()

    classfication = classification_report(test_samples_gt_cpu[test_label_mask_cpu], predict[test_label_mask_cpu] + 1,digits=4)
    kappa = cohen_kappa_score(test_samples_gt_cpu[test_label_mask_cpu], predict[test_label_mask_cpu] + 1)
    print(classfication)
    print("kappa", kappa)
    _,total_indices = utils.sampling(1,data_gt)
    # # print(len(total_indices))
    utils.generate_png(net,net_input,data_gt,device,total_indices,'/home/project/HyperSINet/classification_maps/')
    del net