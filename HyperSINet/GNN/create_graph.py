import numpy as np

def get_label(gt_reshape,train_index,val_index,test_index):
    train_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(train_index)):
        train_samples_gt[train_index[i]] = gt_reshape[train_index[i]]

    test_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(test_index)):
        test_samples_gt[test_index[i]] = gt_reshape[test_index[i]]

    val_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(val_index)):
        val_samples_gt[val_index[i]] = gt_reshape[val_index[i]]

    return train_samples_gt,test_samples_gt,val_samples_gt

def label_to_one_hot(gt,class_num):
    one_hot_label = []
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            temp = np.zeros(class_num,dtype=np.float32)
            if gt[i,j] != 0:
                temp[int(gt[i,j]) - 1] = 1
            one_hot_label.append(temp)
    one_hot_label = np.reshape(one_hot_label,[gt.shape[0],gt.shape[1],class_num])
    return one_hot_label

def get_label_mask(train_samples_gt,test_samples_gt,val_samples_gt,data_gt,class_num):
    height,width = data_gt.shape
    #train_set
    train_label_mask = np.zeros([height*width,class_num])
    temp_ones = np.ones([class_num])
    for i in range(height*width):
        if train_samples_gt[i] != 0:
            train_label_mask[i] = temp_ones
    train_label_mask = np.reshape(train_label_mask,[height*width,class_num])

    #test_set
    test_label_mask = np.zeros([height*width,class_num])
    temp_ones = np.ones([class_num])
    for i in range(height*width):
        if test_samples_gt[i] != 0:
            test_label_mask[i] = temp_ones
    test_label_mask = np.reshape(test_label_mask,[height*width,class_num])

    #val_set
    val_label_mask = np.zeros([height * width, class_num])
    temp_ones = np.ones([class_num])
    for i in range(height * width):
        if val_samples_gt[i] != 0:
            val_label_mask[i] = temp_ones
    val_label_mask = np.reshape(val_label_mask, [height * width, class_num])

    return train_label_mask,test_label_mask,val_label_mask