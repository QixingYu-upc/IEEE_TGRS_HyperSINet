import torch
import numpy as np
import matplotlib.pyplot as plt
import spectral as spy
torch.set_printoptions(profile="full")

def compute_loss(network_output:torch.Tensor,train_samples_gt_onehot:torch.Tensor,train_label_mask:torch.Tensor):
    real_labels = train_samples_gt_onehot
    we = -torch.mul(real_labels,torch.log(network_output))#都是[21025,16],乘的时候是逐元素相乘
    we1 = torch.mul(we,train_label_mask)
    pool_cross_entropy = torch.sum(we1)
    return pool_cross_entropy

def evaluate_performance(network_output, train_samples_gt, train_samples_gt_onehot, zeros):
    with torch.no_grad():
        available_label_idx = (train_samples_gt!=0).float()        # 有效标签的坐标,用于排除背景
        available_label_count = available_label_idx.sum()          # 有效标签的个数
        correct_prediction = torch.where(torch.argmax(network_output, 1) == torch.argmax(train_samples_gt_onehot, 1), available_label_idx, zeros).sum()
        OA = correct_prediction.cpu() / available_label_count
        return OA

def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] * 2.0 / dpi,
                        ground_truth.shape[0] * 2.0 / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)
    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)
    return 0


def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([255, 0, 0]) / 255.
        if item == 1:
            y[index] = np.array([0, 255, 0]) / 255.
        if item == 2:
            y[index] = np.array([0, 0, 255]) / 255.
        if item == 3:
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 4:
            y[index] = np.array([0, 255, 255]) / 255.
        if item == 5:
            y[index] = np.array([255, 0, 255]) / 255.
        if item == 6:
            y[index] = np.array([192, 192, 192]) / 255.
        if item == 7:
            y[index] = np.array([128, 128, 128]) / 255.
        if item == 8:
            y[index] = np.array([128, 0, 0]) / 255.
        if item == 9:
            y[index] = np.array([128, 128, 0]) / 255.
        if item == 10:
            y[index] = np.array([0, 128, 0]) / 255.
        if item == 11:
            y[index] = np.array([128, 0, 128]) / 255.
        if item == 12:
            y[index] = np.array([0, 128, 128]) / 255.
        if item == 13:
            y[index] = np.array([0, 0, 128]) / 255.
        if item == 14:
            y[index] = np.array([255, 165, 0]) / 255.
        if item == 15:
            y[index] = np.array([255, 215, 0]) / 255.
        if item == 16:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 17:
            y[index] = np.array([215, 255, 0]) / 255.
        if item == 18:
            y[index] = np.array([0, 255, 215]) / 255.
        if item == -1:
            y[index] = np.array([0, 0, 0]) / 255.
    return y
def sampling(proportion, ground_truth):
    train = {}
    test = {}
    labels_loc = {}
    m = np.max(ground_truth)
    for i in range(m):
        indexes = [
            j for j, x in enumerate(ground_truth.ravel().tolist())
            if x == i + 1
        ]
        np.random.shuffle(indexes)
        labels_loc[i] = indexes
        if proportion != 1:
            nb_val = max(int((1 - proportion) * len(indexes)), 3)
        else:
            nb_val = 0
        train[i] = indexes[:nb_val]
        test[i] = indexes[nb_val:]
    train_indexes = []
    test_indexes = []
    for i in range(m):
        train_indexes += train[i]
        test_indexes += test[i]
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes

def generate_png(net,net_input,data_gt,device,total_indices, path):
    pred_test1 = []
    X = net_input
    X = X.to(device)
    net.eval()
    gt_hsi = data_gt
    output = net(X)
    pred_test1.extend(output.cpu().argmax(axis=1).detach().numpy())
    gt = gt_hsi.flatten()
    x1_label = np.zeros(gt.shape)
    y1 = np.zeros(gt.shape)
    for i in range(len(gt)):
        if gt[i] == 0:
            gt[i] = 17
            x1_label[i] = 16
        else:
            y1[i] = pred_test1[i]
    gt = gt[:] - 1##0-15
    x1_label[total_indices] = y1[total_indices]
    x1 = np.ravel(x1_label).astype(int)#多维数组变为一维数组
    y_list1 = list_to_colormap(x1)
    y_gt = list_to_colormap(gt)
    y_re1 = np.reshape(y_list1, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
    # gt_re = np.reshape(y_gt, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
    classification_map(y_re1, gt_hsi, 300,path + 'gcnhu.png')
    print('------Get classification maps successful-------')