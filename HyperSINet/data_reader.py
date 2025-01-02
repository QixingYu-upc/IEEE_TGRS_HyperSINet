import numpy as np
import scipy.io as sio
import os
import spectral as spy
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA

class DataReader():
    def __init__(self):
        self.data_cube = None
        self.g_truth = None
    @property#这个东西类似于C++中类的私有成员，在外部函数调用时，无法改变成员的属性
    def cube(self):
        return self.data_cube
    @property
    def truth(self):
        return self.g_truth
    @property
    def normal_cube(self):
        return (self.data_cube - np.min(self.data_cube)) / (np.max(self.data_cube) - np.min(self.data_cube))

class IndianRaw(DataReader):
    def __init__(self):
        super(IndianRaw,self).__init__()
        raw_data_package = sio.loadmat(r"D:\\yqx\\Practice\\NL_GNN_for_HSI\\Datasets\\Indian Pines/Indian_pines_corrected.mat")
        self.data_cube = raw_data_package["data"].astype(np.float32)
        truth = sio.loadmat(r"D:\\yqx\\Practice\\NL_GNN_for_HSI\\Datasets\\Indian Pines/Indian_pines_gt.mat")
        self.g_truth = truth["groundT"].astype(np.float32)
class PaviaURaw(DataReader):
    def __init__(self):
        super(PaviaURaw, self).__init__()
        raw_data_package = sio.loadmat(r"D:\\rmdmy\\datasets\\Pavia\\paviaU.mat")
        self.data_cube = raw_data_package["paviaU"].astype(np.float32)
        truth = sio.loadmat(r"D:\\rmdmy\\datasets\\Pavia\\paviaU_gt.mat")
        # print(truth.keys())
        self.g_truth = truth["Data_gt"].astype(np.float32)

class Salinas(DataReader):
    def __init__(self):
        super(Salinas, self).__init__()
        raw_data_package = sio.loadmat(r"D:\\rmdmy\\datasets\\Salinas\\salinas.mat")
        self.data_cube = raw_data_package["HSI_original"].astype(np.float32)
        truth = sio.loadmat(r"D:\\rmdmy\\datasets\\Salinas\\salinas_gt.mat")
        # print(raw_data_package.keys())
        # print(truth.keys())
        self.g_truth = truth["Data_gt"].astype(np.float32)
class WHU(DataReader):
    def __init__(self):
        super(WHU,self).__init__()
        raw_data_package = sio.loadmat(r"/home/project/GNN_for_HSI/NL_GNN_for_HSI/Datasets/WHU_Hi_LongKou/WHU_Hi_LongKou.mat")
        self.data_cube = raw_data_package["WHU_Hi_LongKou"].astype(np.float32)
        truth = sio.loadmat(r"/home/project/GNN_for_HSI/NL_GNN_for_HSI/Datasets/WHU_Hi_LongKou/WHU_Hi_LongKou_gt.mat")
        self.g_truth = truth["WHU_Hi_LongKou_gt"].astype(np.float32)
class Houston(DataReader):
    def __init__(self):
        super(Houston,self).__init__()
        raw_data_package = sio.loadmat(r"D:\\rmdmy\\datasets\\HoustonU\\Houston.mat")
        self.data_cube = raw_data_package["Houston"].astype(np.float32)
        truth = sio.loadmat(r"D:\\rmdmy\\datasets\\HoustonU\\Houston_gt.mat")
        self.g_truth = truth["Houston_gt"].astype(np.float32)
class KSC(DataReader):
    def __init__(self):
        super(KSC,self).__init__()
        raw_data_package = sio.loadmat(r"D:\\rmdmy\\datasets\\KSC\\KSC.mat")
        self.data_cube = raw_data_package["KSC"].astype(np.float32)
        truth = sio.loadmat(r"D:\\rmdmy\\datasets\\KSC\\KSC_gt.mat")
        # print(raw_data_package.keys())
        # print(truth.keys())
        self.g_truth = truth["KSC_gt"].astype(np.float32)
class Botswana(DataReader):
    def __init__(self):
        super(Botswana,self).__init__()
        raw_data_package = sio.loadmat(r"D:\\rmdmy\\datasets\\Botswana\\Botswana.mat")
        self.data_cube = raw_data_package["Botswana"].astype(np.float32)
        truth = sio.loadmat(r"D:\\rmdmy\\datasets\\Botswana\\Botswana_gt.mat")
        self.g_truth = truth["Botswana_gt"].astype(np.float32)
# PCA
def apply_PCA(data, num_components=75):
    new_data = np.reshape(data, (-1, data.shape[2]))
    pca = PCA(n_components=num_components, whiten=True)
    new_data = pca.fit_transform(new_data)
    new_data = np.reshape(new_data, (data.shape[0], data.shape[1], num_components))
    return new_data, pca


def data_info(train_label=None, val_label=None, test_label=None, start=1):
    class_num = np.max(train_label.astype('int32'))
    if train_label is not None and val_label is not None and test_label is not None:

        total_train_pixel = 0
        total_val_pixel = 0
        total_test_pixel = 0
        train_mat_num = Counter(train_label.flatten())
        val_mat_num = Counter(val_label.flatten())
        test_mat_num = Counter(test_label.flatten())

        for i in range(start, class_num + 1):
            print("class", i, "\t", train_mat_num[i], "\t", val_mat_num[i], "\t", test_mat_num[i])
            total_train_pixel += train_mat_num[i]
            total_val_pixel += val_mat_num[i]
            total_test_pixel += test_mat_num[i]
        print("total", "    \t", total_train_pixel, "\t", total_val_pixel, "\t", total_test_pixel)

    elif train_label is not None and val_label is not None:

        total_train_pixel = 0
        total_val_pixel = 0
        train_mat_num = Counter(train_label.flatten())
        val_mat_num = Counter(val_label.flatten())

        for i in range(start, class_num + 1):
            print("class", i, "\t", train_mat_num[i], "\t", val_mat_num[i])
            total_train_pixel += train_mat_num[i]
            total_val_pixel += val_mat_num[i]
        print("total", "    \t", total_train_pixel, "\t", total_val_pixel)

    elif train_label is not None:
        total_pixel = 0
        data_mat_num = Counter(train_label.flatten())

        for i in range(start, class_num + 1):
            print("class", i, "\t", data_mat_num[i])
            total_pixel += data_mat_num[i]
        print("total:   ", total_pixel)

    else:
        raise ValueError("labels are None")


def draw(label, name: str = "default", scale: float = 4.0, dpi: int = 400, save_img=True):
    fig, ax = plt.subplots()
    numlabel = np.array(label)
    v = spy.imshow(classes=numlabel.astype(np.int16), fignum=fig.number)
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
    foo_fig = plt.gcf()  # 'get current figure'
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    if save_img:
        foo_fig.savefig(name + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0)


# if __name__ == "__main__":
#     data = PaviaURaw().cube
#     data_gt = PaviaURaw().truth
#     print(data)
#     data_info(data_gt)
#     draw(data_gt, save_img=1)
#     print(data.keys())
#     print(data_gt.keys())