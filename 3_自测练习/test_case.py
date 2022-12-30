#!/usr/bin/env python3

from my_solution import solution


# 测试用例
def test_solution():
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

    test(gnn, gen, args.n_classes)

    print ('total num of params:', count_parameters(gnn))
