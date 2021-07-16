from numpy import *
import scipy.io as sio
import numpy as np
import os
from sklearn import preprocessing
def load_data(args):
    db_name = args.data_dir
    matcontent = sio.loadmat('/home/huangchong/2021_work/EPGN/data' + '/' + db_name + '/' + 'data.mat')
    # 以下维度均以AwA2数据集为例
    train_att = matcontent['att_train'] # 23527, 85
    seen_pro = matcontent['seen_pro']   # 40, 85
    attribute = matcontent['attribute'] # 50, 85
    unseen_pro = matcontent['unseen_pro'] # 10, 85

    train_fea = matcontent['train_fea']
    test_seen_fea = matcontent['test_seen_fea']
    test_unseen_fea = matcontent['test_unseen_fea']

    if args.preprocess:
        scaler = preprocessing.MinMaxScaler()
        _train_feature = scaler.fit_transform(train_fea)
        _test_seen_feature = scaler.transform(test_seen_fea)
        _test_unseen_feature = scaler.transform(test_unseen_fea)
        mx = _train_feature.max()
        train_fea = train_fea*(1 / mx)
        test_seen_fea = test_seen_fea*(1 /mx)
        test_unseen_fea= test_unseen_fea*(1/mx)
    # TODO 这和上面不是重复了吗？这不就相当于两次归一化了嘛
    scaler = preprocessing.MinMaxScaler()
    _train_feature = scaler.fit_transform(train_fea)
    _test_seen_feature = scaler.transform(test_seen_fea)
    _test_unseen_feature = scaler.transform(test_unseen_fea)
    mx = _train_feature.max()
    train_fea = train_fea*(1 / mx)
    test_seen_fea = test_seen_fea*(1 /mx)
    test_unseen_fea= test_unseen_fea*(1/mx)
    # 因为数据文件中的下标是从1开始的，因此减1，使得从0开始
    matcontent = sio.loadmat('/home/huangchong/2021_work/EPGN/data' + "/" + db_name + "/" + "label.mat")
    train_idx = matcontent['train_idx'] -1
    train_label = matcontent['train_label_new'] # one-hot形式的
    test_unseen_idex = matcontent['test_unseen_idex']-1
    test_seen_idex = matcontent['test_seen_idex']-1
    data = {
        'attribute': attribute,
        'train_fea': train_fea,
        'train_att': train_att,
        'train_idx': train_idx,
        'seen_pro':seen_pro,
        'train_lab':train_label,
        'test_seen_fea': test_seen_fea,
        'test_unseen_fea': test_unseen_fea,
        'unseen_pro': unseen_pro,
        'test_unseen_idex': test_unseen_idex,
        'test_seen_idex': test_seen_idex
    }
    return data

def prepare_data(data, args):
    seen_fea = data['train_fea']
    seen_att = data['train_att']
    seen_lab = data['train_lab']
    seen_idx = data['train_idx']

    seen_cla_num = data['seen_pro'].shape[0]
    unseen_cla_num = data['unseen_pro'].shape[0]

    # TODO 这里的train, test是否指的是episode中对base model的训练
    train_cla_num = seen_cla_num - args.selected_cla_num
    test_cla_num = args.selected_cla_num

    seen_cla_idx = list(range(seen_cla_num)) # [0,1,..,39]
    random.shuffle(seen_cla_idx)
    '''
        从训练类里面随机选择10个作为episode的test
    '''
    train_cla_idx = seen_cla_idx[:train_cla_num]
    test_cla_idx = seen_cla_idx[train_cla_num:seen_cla_num]
    '''
        训练类中用来train base model的一共30个类
        从训练类挑出episode中用来train的样本下标，存到meta_list
        meta_data即episode中用来trian的样本
    '''    
    meta_list = []
    for i in range(train_cla_num):
        temp = np.where(seen_idx == train_cla_idx[i])
        index = temp[0]
        meta_list.extend(index)
    meta_train_fea = seen_fea[meta_list]
    meta_train_att = seen_att[meta_list]
    '''
        meta-train时使用的lab不是one-hot形式的，因为要传入交叉熵损失
    '''
    # meta_train_lab = seen_lab[meta_list]
    meta_train_lab = seen_idx[meta_list]
    meta_data = {
        'meta_train_fea':meta_train_fea,
        'meta_train_att':meta_train_att,
        'meta_train_lab':meta_train_lab,
        'meta_train_pro':data['seen_pro'],
    }
    '''
        训练类中用来test的一共10个类(i.e.selected class num)
    '''
    learner_list = []
    for j in range(test_cla_num):
        temp = np.where(seen_idx == test_cla_idx[j])
        index = temp[0]
        learner_list.extend(index)

    learner_test_fea = seen_fea[learner_list]
    learner_test_att = seen_att[learner_list]
    learner_test_lab = seen_lab[learner_list]
    learner_test_idx = seen_idx[learner_list]
    learner_test_pro = data['seen_pro']
    learner_data = {
        'learner_test_fea':learner_test_fea,
        'learner_test_att':learner_test_att,
        'learner_test_lab':learner_test_lab,
        'learner_test_idx':learner_test_idx,
        'learner_test_pro':learner_test_pro,
    }
    # learner_data = {
    #     'learner_test_fea':learner_test_fea[:4000],
    #     'learner_test_att':learner_test_att[:4000],
    #     'learner_test_lab':learner_test_lab[:4000],
    #     'learner_test_idx':learner_test_idx[:4000],
    #     'learner_test_pro':learner_test_pro,
    # }

    return meta_data, learner_data

def get_batch(data, batch_size):
    '''
        可重构成dataloader
    '''
    img = data['meta_train_fea']
    att = data['meta_train_att']
    pro = data['meta_train_pro']
    cla = data['meta_train_lab']

    while True:
        idx = np.arange(0, len(img))
        np.random.shuffle(idx)
        shuf_visual = img[idx]
        shuf_attr = att[idx]
        shuf_cla = cla[idx]

        for batch_index in range(0, len(img), batch_size):
            visual_batch = shuf_visual[batch_index:batch_index + batch_size]
            visual_batch = visual_batch.astype("float32")
            attr_batch = shuf_attr[batch_index:batch_index + batch_size]
            cla_batch = shuf_cla[batch_index:batch_index + batch_size]
            yield visual_batch, attr_batch, cla_batch, pro

def get_learner_data(data):
    test_fea = data['learner_test_fea']
    test_lab = data['learner_test_lab']
    # test_lab = data['learner_test_idx']
    test_pro = data['learner_test_pro']
    return test_fea, test_pro, test_lab

# TODO 这个程序中貌似没用到s
def data_iterator(data, selected_sum):
    x = data['train_fea']
    train_idex = data['train_idex']
    unique_tr_label = np.unique(train_idex)
    batch_att_pro = data['seen_pro']
    batch_fea_pro = np.zeros(shape=(len(unique_tr_label), 2048))

    for i in range(len(unique_tr_label)):
        temp = np.where(train_idex == unique_tr_label[i])
        index = temp[0]
        idxs = np.arange(0, len(index))
        np.random.shuffle(idxs)
        selected_idx = idxs[0:selected_sum]
        selected_fea = x[index[selected_idx]]
        mean_fea = mean(selected_fea, 0)
        batch_fea_pro[i] = mean_fea
    return batch_att_pro, batch_fea_pro


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir