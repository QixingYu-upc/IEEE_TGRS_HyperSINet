import numpy as np

def split_data(gt_reshape,class_num,train_ratio,val_ratio,train_num,val_num,samples_type):
    train_index = []
    test_index = []
    val_index = []
    if samples_type == 'ratio':
        for i in range(class_num):
            idx = np.where(gt_reshape == i+1)[-1]
            samplesCount = len(idx)
            train_num = np.ceil(samplesCount * train_ratio).astype('int32')
            val_num = np.ceil(samplesCount * val_ratio).astype('int32')
            np.random.shuffle(idx)
            train_index.append(idx[:train_num])
            val_index.append(idx[train_num:train_num+val_num])
            test_index.append(idx[train_num+val_num:])
    else:
        sample_num = train_num
        for i in range(class_num):
            idx = np.where(gt_reshape == i+1)[-1]
            samplesCount = len(idx)
            max_index = np.max(samplesCount) + 1
            np.random.shuffle(idx)
            if sample_num > max_index:
                sample_num = 10
            else:
                sample_num = train_num
            train_index.append(idx[:sample_num])
            val_index.append(idx[sample_num:sample_num+class_num])
            test_index.append(idx[sample_num+class_num:])

    train_index = np.concatenate(train_index,axis=0)
    val_index = np.concatenate(val_index,axis=0)
    test_index = np.concatenate(test_index,axis=0)

    return train_index,val_index,test_index
