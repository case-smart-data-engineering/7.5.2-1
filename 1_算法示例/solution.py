#!/usr/bin/env python3


import numpy as np
import os
from data_generator import Generator
from load import get_gnn_inputs
from models import GNN_multiclass
import time
import argparse

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from losses import compute_loss_multiclass, compute_accuracy_multiclass

parser = argparse.ArgumentParser()

###############################################################################
#                             General Settings                                #
###############################################################################

#参数设置
parser.add_argument('--num_examples_train', nargs='?', const=1, type=int,
                    default=10)
parser.add_argument('--num_examples_test', nargs='?', const=1, type=int,
                    default=10)
parser.add_argument('--edge_density', nargs='?', const=1, type=float,
                    default=0.2)
parser.add_argument('--p_SBM', nargs='?', const=1, type=float,
                    default=0.0)
parser.add_argument('--q_SBM', nargs='?', const=1, type=float,
                    default=0.045)
parser.add_argument('--random_noise', action='store_true')
parser.add_argument('--noise', nargs='?', const=1, type=float, default=0.03)
parser.add_argument('--noise_model', nargs='?', const=1, type=int, default=2)
parser.add_argument('--generative_model', nargs='?', const=1, type=str,
                    default='SBM_multiclass')
parser.add_argument('--batch_size', nargs='?', const=1, type=int, default=1)
parser.add_argument('--mode', nargs='?', const=1, type=str, default='train')
parser.add_argument('--path_gnn', nargs='?', const=1, type=str, default='')
parser.add_argument('--filename_existing_gnn', nargs='?', const=1, type=str, default='')
parser.add_argument('--print_freq', nargs='?', const=1, type=int, default=1)
parser.add_argument('--test_freq', nargs='?', const=1, type=int, default=500)
parser.add_argument('--save_freq', nargs='?', const=1, type=int, default=2000)
parser.add_argument('--clip_grad_norm', nargs='?', const=1, type=float,
                    default=40.0)
parser.add_argument('--freeze_bn', dest='eval_vs_train', action='store_true')
parser.set_defaults(eval_vs_train=False)

###############################################################################
#                                 GNN Settings                                #
###############################################################################

parser.add_argument('--num_features', nargs='?', const=1, type=int,
                    default=10)
parser.add_argument('--num_layers', nargs='?', const=1, type=int,
                    default=1)
parser.add_argument('--n_classes', nargs='?', const=1, type=int,
                    default=2)
parser.add_argument('--J', nargs='?', const=1, type=int, default=2)
parser.add_argument('--N_train', nargs='?', const=1, type=int, default=10)
parser.add_argument('--N_test', nargs='?', const=1, type=int, default=10)
parser.add_argument('--lr', nargs='?', const=1, type=float, default=0.004)

args = parser.parse_args()

#判断cuda是否可以使用
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor    #GPU的张量类型
    dtype_l = torch.cuda.LongTensor
    # torch.cuda.manual_seed(0)
else:
    dtype = torch.FloatTensor   #CPU上的数据类型
    dtype_l = torch.LongTensor
    # torch.manual_seed(1)

batch_size = args.batch_size
#交叉熵损失函数
criterion = nn.CrossEntropyLoss()
#template将字符串的格式固定下来，重复利用。常用于自动生成测试用例
template1 = '{:<10} {:<10} {:<10} {:<15} {:<10} {:<10} {:<10} '
template2 = '{:<10} {:<10.5f} {:<10.5f} {:<15} {:<10} {:<10} {:<10.3f} \n'
template3 = '{:<10} {:<10} {:<10} '
template4 = '{:<10} {:<10.5f} {:<10.5f} \n'

#训练单个用例
def train_single(gnn, optimizer, gen, n_classes, it):
    #返回当前时间的时间戳
    start = time.time()
    #将训练用例生成图模型
    W, labels = gen.sample_otf_single(is_training=True, cuda=torch.cuda.is_available())
    #类型转换，使用dtype_l类型
    labels = labels.type(dtype_l)
    #对标签数据进行折半处理
    if (args.generative_model == 'SBM_multiclass') and (args.n_classes == 2):
        labels = (labels + 1)/2
    #矩阵归一化处理
    WW, x = get_gnn_inputs(W, args.J)

    if (torch.cuda.is_available()):
        WW.cuda()
        x.cuda()

    pred = gnn(WW.type(dtype), x.type(dtype))
    #得出多类损失结果
    loss = compute_loss_multiclass(pred, labels, n_classes)
    #使用backward函数前需要将梯度清零
    gnn.zero_grad() 
    #调用backward函数
    loss.backward()

    #梯度裁剪，每次都需要与clip_grad_norm相乘  Parameter函数对某个张量进行参数化
    nn.utils.clip_grad_norm_(gnn.parameters(), args.clip_grad_norm)
    #将optimizer的图形设计为水平基线，数据点通过垂直基线连接到该基线
    optimizer.step()
    #得到多类输出结果
    acc = compute_accuracy_multiclass(pred, labels, n_classes)
    #计算运行时长
    elapsed = time.time() - start

    #将tensor转成numpy
    if(torch.cuda.is_available()):
        loss_value = float(loss.data.cpu().numpy())
    else:
        loss_value = float(loss.data.numpy())

    #编号，损失，输出，边缘密度，噪声，模型，运行时间
    info = ['iter', 'avg loss', 'avg acc', 'edge_density',
            'noise', 'model', 'elapsed']
    out = [it, loss_value, acc, args.edge_density,
           args.noise, 'GNN', elapsed]
    print(template1.format(*info))
    print(template2.format(*out))

    del WW
    del x

    return loss_value, acc

#训练模型
def train(gnn, gen, n_classes=args.n_classes, iters=args.num_examples_train):
    gnn.train()
    #优化算法
    optimizer = torch.optim.Adamax(gnn.parameters(), lr=args.lr)
    loss_lst = np.zeros([iters])
    acc_lst = np.zeros([iters])
    for it in range(iters):
        loss_single, acc_single = train_single(gnn, optimizer, gen, n_classes, it)#计算单个用例的损失和得出单个用例的输出
        loss_lst[it] = loss_single  #损失值列表
        acc_lst[it] = acc_single    #输出值列表
        torch.cuda.empty_cache()
    # mean：取均值  std：标准差计算
    print ('Avg train loss', np.mean(loss_lst))
    print ('Avg train acc', np.mean(acc_lst))
    print ('Std train acc', np.std(acc_lst))


#测试单个用例
def test_single(gnn, gen, n_classes, it):
    #返回当前时间的时间戳
    start = time.time()
    #将测试用例生成图模型
    W, labels = gen.sample_otf_single(is_training=False, cuda=torch.cuda.is_available())
    #类型转换，使用dtype_l类型
    labels = labels.type(dtype_l)
    #对标签数据进行折半处理
    if (args.generative_model == 'SBM_multiclass') and (args.n_classes == 2):
        labels = (labels + 1)/2
    #矩阵归一化处理
    WW, x = get_gnn_inputs(W, args.J)

    print ('WW', WW.shape)    #读取矩阵的长度

    if (torch.cuda.is_available()):
        WW.cuda()
        x.cuda()

    pred_single = gnn(WW.type(dtype), x.type(dtype))
    labels_single = labels
    #得出多类损失结果
    loss_test = compute_loss_multiclass(pred_single, labels_single, n_classes)
    #得到多类输出结果
    acc_test = compute_accuracy_multiclass(pred_single, labels_single, n_classes)
    #计算运行时长
    elapsed = time.time() - start
    #将tensor转成numpy
    if(torch.cuda.is_available()):
        loss_value = float(loss_test.data.cpu().numpy())
    else:
        loss_value = float(loss_test.data.numpy())
    
    #编号，损失，输出，边缘密度，噪声，模型，运行时间
    info = ['iter', 'avg loss', 'avg acc', 'edge_density',
            'noise', 'model', 'elapsed']
    out = [it, loss_value, acc_test, args.edge_density,
           args.noise, 'GNN', elapsed]
    print(template1.format(*info))
    print(template2.format(*out))

    del WW
    del x

    return loss_value, acc_test


#测试模型
def test(gnn, gen, n_classes, iters=args.num_examples_test):
    gnn.train()
    loss_lst = np.zeros([iters])
    acc_lst = np.zeros([iters])
    for it in range(iters):
        loss_single, acc_single = test_single(gnn, gen, n_classes, it)  #计算单个用例的损失和得出单个用例的输出
        loss_lst[it] = loss_single  #损失值列表
        acc_lst[it] = acc_single    #输出值列表
        torch.cuda.empty_cache()
    # mean：取均值  std：标准差计算
    print ('Avg test loss', np.mean(loss_lst))   
    print ('Avg test acc', np.mean(acc_lst))
    print ('Std test acc', np.std(acc_lst))

#统计模型参数总数
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



if __name__ == '__main__':

    gen = Generator()
    gen.N_train = args.N_train
    gen.N_test = args.N_test
    gen.edge_density = args.edge_density
    gen.p_SBM = args.p_SBM
    gen.q_SBM = args.q_SBM
    gen.random_noise = args.random_noise
    gen.noise = args.noise
    gen.noise_model = args.noise_model
    gen.generative_model = args.generative_model
    gen.n_classes = args.n_classes


    torch.backends.cudnn.enabled=False
    
    #测试模型
    if (args.mode == 'test'):
        print ('In testing mode')
        filename = args.filename_existing_gnn
        path_plus_name = os.path.join(args.path_gnn, filename)
        if ((filename != '') and (os.path.exists(path_plus_name))):
            print ('Loading gnn ' + filename)
            gnn = torch.load(path_plus_name)
            if torch.cuda.is_available():
                gnn.cuda()
        else:
            print ('No such a gnn exists; creating a brand new one')
            if (args.generative_model == 'SBM_multiclass'):
                gnn = GNN_multiclass(args.num_features, args.num_layers, args.J + 2, n_classes=args.n_classes)
            filename = 'gnn_J' + str(args.J) + '_lyr' + str(args.num_layers) + '_Ntr' + str(args.N_train) + '_num' + str(args.num_examples_train)
            path_plus_name = os.path.join(args.path_gnn, filename)
            if torch.cuda.is_available():
                gnn.cuda()
            print ('Training begins')

    #训练模型
    elif (args.mode == 'train'):
        filename = args.filename_existing_gnn
        path_plus_name = os.path.join(args.path_gnn, filename)
        if ((filename != '') and (os.path.exists(path_plus_name))):
            print ('Loading gnn ' + filename)
            gnn = torch.load(path_plus_name)
            filename = filename + '_Ntr' + str(args.N_train) + '_num' + str(args.num_examples_train)
            path_plus_name = os.path.join(args.path_gnn, filename)
        else:
            print ('No such a gnn exists; creating a brand new one')
            filename = 'gnn_J' + str(args.J) + '_lyr' + str(args.num_layers) + '_Ntr' + str(args.N_train) + '_num' + str(args.num_examples_train)
            path_plus_name = os.path.join(args.path_gnn, filename)
            if (args.generative_model == 'SBM_multiclass'):
                gnn = GNN_multiclass(args.num_features, args.num_layers, args.J + 2, n_classes=args.n_classes)

        print ('total num of params:', count_parameters(gnn))

        if torch.cuda.is_available():
            gnn.cuda()
        print ('Training begins')
        if (args.generative_model == 'SBM_multiclass'):
            train(gnn, gen, args.n_classes)
        print ('Saving gnn ' + filename)
        if torch.cuda.is_available():
            torch.save(gnn.cpu(), path_plus_name)
            gnn.cuda()
        else:
            torch.save(gnn, path_plus_name)


    print ('Testing the GNN:')
    if args.eval_vs_train:
        print ('model status: eval')
        gnn.eval()
    else:
        print ('model status: train')
        gnn.train()
    #测试用例
    test(gnn, gen, args.n_classes)
    #输出参数的总数
    print ('total num of params:', count_parameters(gnn))


