#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :awa1.py
@Date    :2021/04/15 20:29:49
@Author  :Chong
    pytorch实现EPGN
'''
import argparse
from utils import *
import torch
import torch.nn as nn
import time
from ops import *
from test import *
import os
import itertools
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of prototype generating network for ZSL"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data_dir',type=str, default='CUB', help='[AwA1 / AwA2 / CUB / FLO]')
    parser.add_argument('--img_dim', type=int, default=2048, help='the image dimension')
    parser.add_argument('--hid_dim', type=int, default=1800, help='the hidden dimension, default: 1600')
    parser.add_argument('--mid_dim', type=int, default=1600, help='the middle dimension of discriminator, default: 1800')
    parser.add_argument('--att_dim', type=int, default=1024, help='the attribute dimension, AwA: 85, CUB: 1024,FLO: 1024')
    parser.add_argument('--cla_num', type=int, default=200, help='the class number')
    parser.add_argument('--tr_cla_num', type=int, default=150, help='the training class number')
    parser.add_argument('--selected_cla_num',type=int, default=10, help='the selected class number for meta-test')
    parser.add_argument('--lr', type=float32, default=5e-5, help='the learning rate, default: 1e-4')
    parser.add_argument('--preprocess', action='store_true', default=False, help='MaxMin process')
    parser.add_argument('--dropout',action='store_true',default=False, help='enable dropout')
    parser.add_argument('--epoch', type=int, default=15, help='the max iterations, default: 5000')
    parser.add_argument('--episode',type=int, default=100, help='the max iterations of episodes')
    parser.add_argument('--inner_loop',type=int, default=20, help='the inner loop')
    parser.add_argument('--batch_size', type=int, default=20, help='the batch_size, default: 100')
    parser.add_argument('--manualSeed', type=int, default=4198, help='maunal seed') # 4198
    return parser.parse_args()

class G(nn.Module):
    '''
        Semantic -> Visual prototype
    '''
    def __init__(self, args):
        super(G, self).__init__()
        self.modelG = nn.Sequential(
            nn.Linear(args.att_dim, args.hid_dim),
            nn.Tanh(),
            nn.Linear(args.hid_dim, args.img_dim),
            nn.ReLU()
        )
    def forward(self, x):
        out = self.modelG(x)
        return out

class F(nn.Module):
    '''
        V -> S
    '''
    def __init__(self, args):
        super(F, self).__init__()
        if args.dropout:
            self.modelF = nn.Sequential(
            nn.Linear(args.img_dim, args.hid_dim),
            nn.Tanh(),
            nn.Dropout(0.8),
            nn.Linear(args.hid_dim, args.att_dim),
            nn.ReLU()
            )
        # else:
        #     self.modelF = nn.Sequential(
        #         nn.Linear(args.img_dim, args.hid_dim),
        #         nn.Tanh(),
        #         nn.Linear(args.hid_dim, args.att_dim),
        #         nn.ReLU()
        #     )
    def forward(self, x):
        out = self.modelF(x)
        return out
class D(nn.Module):
    def __init__(self, args):
        super(D, self).__init__()
        self.modelD = nn.Sequential(
            nn.Linear(args.img_dim+args.att_dim, args.hid_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(args.hid_dim, 1),
            nn.ReLU()
            )
    def forward(self, x, c):
        inputs = torch.cat((x,c),dim=1)
        out = self.modelD(inputs)
        return out
def truncated_normal_(tensor,mean=0,std=0.09):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size+(4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor
'''
    参照官方tf代码的初始化方式
'''
def weights_init(m):
    if isinstance(m, nn.Linear):
        truncated_normal_(m.weight, std=0.01)
        nn.init.constant_(m.bias, 0.01)
def test_seen(data, args):
    attribute = data['attribute']
    test_feats = data['test_seen_fea'] #4958 x 2048
    test_idex = data['test_seen_idex']
    test_id = np.arange(args.cla_num) # gzsl设置下，对于可见类和不可见类上准确率的计算要考虑全部类别。
    test_attr = attribute[test_id]
    att_pre = netG(torch.from_numpy(test_attr).float().cuda())
    att_pre = att_pre.cpu().detach().numpy()
    # TODO 重新写一下准确率计算，原作者写的有些复杂，简化一下
    acc = compute_accuracy(att_pre, test_feats, test_idex, test_id)
    return acc
def test_unseen(data, args):
    attribute = data['attribute']
    test_feats = data['test_unseen_fea']  # 5685 x 2048
    test_idex = data['test_unseen_idex']
    test_id = np.arange(args.cla_num) # gzsl设置下，对于可见类和不可见类上准确率的计算要考虑全部类别。
    test_attr = attribute[test_id]
    att_pre = netG(torch.from_numpy(test_attr).float().cuda())
    att_pre = att_pre.cpu().detach().numpy()
    acc = compute_accuracy(att_pre, test_feats, test_idex,test_id)
    return acc
def test_zsl(data):
    
    start_time = time.time()
    attribute = data['attribute']
    test_feats = data['test_unseen_fea']
    test_idex = data['test_unseen_idex']
    test_id = np.unique(test_idex)
    test_attr = attribute[test_id]
    att_pre = netG(torch.from_numpy(test_attr).float().cuda())
    att_pre = att_pre.cpu().detach().numpy()
    acc = compute_accuracy(att_pre, test_feats, test_idex, test_id)
    return acc
if __name__ == '__main__':
    args = parse_args()
    '''
        打印一些超参数信息
    '''
    print("###### Information #######")
    print('# batch_size:', args.batch_size)
    print('# epoch_number:', args.epoch)
    if args is None:
        exit()
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    torch.cuda.manual_seed(args.manualSeed)
    torch.cuda.manual_seed_all(args.manualSeed)
    # TODO pytorch的一些种子设置
    netG = G(args).cuda()
    netF = F(args).cuda()
    netD = D(args).cuda()
    netG.apply(weights_init)
    netF.apply(weights_init)
    netD.apply(weights_init)
    # b_optimizer = torch.optim.Adam([{'params':netG.parameters}
    #                                 ,{'params':netF.parameters}], lr=args.lr)
    b_optimizer = torch.optim.Adam(itertools.chain(netG.parameters(), netF.parameters()), lr=args.lr)
    d_optimizer = torch.optim.Adam(netD.parameters(), lr=args.lr)
    # TODO 这里还会用到netD的参数吗
    g_optimizer = torch.optim.Adam(netG.parameters(), lr=args.lr)
    c_optimizer = torch.optim.Adam(netG.parameters(), lr=args.lr/10)
    '''
        meta train
    '''
    start_time = time.time()
    data = load_data(args)
    best_zsl = 0
    best_gzsl = 0
    best_epoch = -1
    for epo in range(args.epoch):
        meta_data, learner_data = prepare_data(data, args)
        netG.train()
        netF.train()
        netD.train()
        for episode in range(args.episode):
            train_loader = get_batch(meta_data, args.batch_size) # 这里train_loader是个generator
            img_batch, att_batch, cla_batch, train_pro = next(train_loader)
            img_batch, att_batch, cla_batch, train_pro = torch.from_numpy(img_batch).float(), torch.from_numpy(att_batch).float(), torch.from_numpy(cla_batch).long().squeeze(), torch.from_numpy(train_pro).float()
            img_batch, att_batch, cla_batch, train_pro = img_batch.cuda(), att_batch.cuda(), cla_batch.cuda(), train_pro.cuda()
            # print('att_batch:', att_batch.shape) # (bs, 85)
            '''
                cla_batch: (bs, 1)  img_batch: (bs, 2048)
                train_pro: (40, 85)
                att_output_pro: (40, 2048)
                torch.mm(img_batch, att_output_pro.t()): (bs, 40)
            '''
            pre_img = netG(att_batch)
            pre_att = netF(img_batch)
            att_output_pro = netG(train_pro) # 即将类语义特征映射为类视觉原型
            '''
                1.更新base model(netF和netG)
            '''
            criterion = nn.CrossEntropyLoss()
            # print('cla_batch.shape:', cla_batch.shape, cla_batch)
            cla_loss = criterion(torch.mm(img_batch, att_output_pro.t()), cla_batch) + criterion(torch.mm(pre_att, train_pro.t()), cla_batch)
            lse = nn.MSELoss(reduction='mean')
            # lse_loss = torch.mean(torch.quare())
            lse_loss = lse(img_batch, pre_img) + lse(att_batch, pre_att)
            b_optimizer.zero_grad()
            d_optimizer.zero_grad()
            b_loss = lse_loss + cla_loss
            b_loss.backward()
            '''
                2.更新netD
            '''
            # d_optimizer.zero_grad()
            d_image_real = netD(img_batch, att_batch)
            # d_image_fake = netD(pre_img, pre_att)
            d_image_fake = netD(pre_img.detach(), pre_att.detach())
            d_loss = discriminator_loss('wgan', d_image_real, d_image_fake)
            d_loss.backward()
            d_optimizer.step()
            b_optimizer.step()
            '''
                3.更新netG
            '''
            g_optimizer.zero_grad()
            new_pre_img = netG(att_batch)
            # 下面是使用更新过的D的情况，效果反而不如不用
            # new_pre_att = netF(img_batch)
            # d_image_real = netD(img_batch, att_batch)
            # d_image_fake = netD(new_pre_img, new_pre_att.detach())
            mse = torch.sum((img_batch - new_pre_img) ** 2, 1)
            e_loss = torch.mean(1e-3*mse+1e-3*torch.log(d_image_real.detach()))
            g_loss = generator_loss('wgan', d_image_real.detach(), d_image_fake.detach()) + torch.log(e_loss)
            g_loss.backward()
            g_optimizer.step()
            if (episode + 1) % 10 == 0:
                print ('[epoch {}/{}, episode {}/{}] => loss:{:.5f}'.format(epo+1, args.epoch, episode+1, args.episode, b_loss.item()))
        '''
            meta test，对应论文中的refining model
        '''
        learner_fea, learner_pro, learner_lab = get_learner_data(learner_data)
        learner_fea, learner_pro, learner_lab = torch.from_numpy(learner_fea).float(), torch.from_numpy(learner_pro).float(), torch.from_numpy(learner_lab).long()
        learner_fea, learner_pro, learner_lab = learner_fea.cuda(), learner_pro.cuda(), learner_lab.cuda()
        '''
            将learner_lab由one-hot转成普通的label.即[4894,40]->[4894]
        '''
        learner_lab = learner_lab.argmax(dim=-1)
        for i in range(args.inner_loop):
            
            '''
                c_loss
            '''
            c_optimizer.zero_grad()
            learner_pro_img = netG(learner_pro) # (40, 2048)
            dists = euclidean_distance(learner_pro_img, learner_fea) # (40, bs)
            dists = dists.t() # (bs, 40)
            # log_dists = torch.nn.functional.log_softmax(-dists) # (4894, 40)
            # c_loss = -torch.mean(torch.sum(torch.mul(learner_lab.float(), log_dists), dim=-1).flatten())
            '''
                将c_loss重构成用CrossEntropy的形式的形式
            '''
            criterion = nn.CrossEntropyLoss()
            c_loss = criterion(-dists, learner_lab)
            '''
                排查CUDA out of memory错误
            '''
            # import pynvml
            # pynvml.nvmlInit()
            # handle = pynvml.nvmlDeviceGetHandleByIndex(0) # 0表示显卡标号
            # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            # print('inner loop:{} mem used:{}'.format(i, meminfo.used/1024**2))  #已用显存大小
            c_loss.backward()
            c_optimizer.step()
            '''
                不清除cache，在运行到epoch3时会报错CUDA out of memory。
            '''
            torch.cuda.empty_cache()
        if (epo+1)%1 == 0:
            start_time = time.time()
            '''
                test()
            '''
            netG.eval()
            netF.eval()
            netD.eval()
            '''
                ZSL test
            '''
            acc = test_zsl(data)
            print('Acc: unseen class', acc)
            '''
                GZSL test
            '''
            acc_unseen = test_unseen(data, args)
            acc_seen = test_seen(data, args)
            H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
            if H > best_gzsl:
                best_gzsl = H
                best_zsl = acc
                best_epoch = epo
            print ('accuracy: seen class:', acc_seen, '| unseen class:', acc_unseen, '| harmonic:', H)
            end_time = time.time()
            print('# The test time is: %4.4f' %(end_time - start_time))
    end_time = time.time()
    print('# The training time is: %4.4f' %(end_time - start_time))
    print('best gzsl:{} @ epoch:{}, cur zsl:{}'.format(best_gzsl, best_epoch, best_zsl))


