#!/usr/bin/env python3

from my_solution import *


# 测试用例
def test_solution():
    gen = Generator()
    gen.N_train =10
    gen.N_test =10
    gen.edge_density =0.2
    gen.p_SBM =0.0
    gen.q_SBM =0.045
    gen.noise = 0.03
    gen.noise_model =2
    gen.generative_model ='SBM_multiclass'
    gen.n_classes = 2
    torch.backends.cudnn.enabled=False
    gnn = GNN_multiclass(10, 1, 2 + 2, n_classes=2)
    print ('Testing the GNN:')
    test(gnn, gen, 2)
    print ('total num of params:', count_parameters(gnn))
