from dataset_gcl import Dataset_gcl
from options import get_options
from model import *
import dgl
import pickle
import numpy as np
import os
from MyDataLoader_ud import *
from time import time
import math
import networkx as nx
from random import shuffle
import random

def preprocess(data_path, device, options):

    in_dim = get_options().in_dim
    if not os.path.exists(os.path.join(data_path, 'i{}'.format(options.num_input))):
        os.makedirs(os.path.join(data_path, 'i{}'.format(options.num_input)))
    # data_file = os.path.join(data_path, 'i{}/{}.pkl'.format(options.num_input, options.split))

    # if the dataset for contrastive learning is not ready,
    # then generate the data
    data_file = os.path.join(data_path, 'i{}/origin.pkl'.format(options.num_input))
    if os.path.exists(data_file) is False:
        datapath = "../truthtables/i{}/implementation/".format(options.num_input)
        th.multiprocessing.set_sharing_strategy('file_system')
        dataset = Dataset_gcl(datapath)
        #print(dataset.batch_graph.ndata)
        data_file = os.path.join(data_path, 'i{}/origin.pkl'.format(options.num_input))
        with open(data_file, 'wb') as f:
            pickle.dump((dataset.batch_graph, dataset.POs, dataset.depth), f)
        for i in range(3):
            data_file = os.path.join(data_path, 'i{}/aug{}.pkl'.format(options.num_input,i+1))
            with open(data_file, 'wb') as f:
                pickle.dump((dataset.aug_batch_graphs[i], dataset.aug_POs[i], dataset.aug_depth[i]), f)

    # load/create the model
    if options.pre_train:
        with open(os.path.join(options.pre_model_dir, 'model.pkl'), 'rb') as f:
            _, model,proj_head = pickle.load(f)
    else:
        # a model is composed of a FuncGNN and a projection head
        network = FuncGNN
        in_nlayers = options.in_nlayers
        model = network(
            ntypes=in_dim,
            hidden_dim=options.hidden_dim,
            out_dim=options.out_dim,
            n_layers=in_nlayers,
            dropout=options.gcn_dropout,
        )
        proj_head = Projection_Head(
            in_feats=options.out_dim,
            out_feats=options.out_dim
        ).to(device)
    print("creating model in:", options.model_saving_dir)
    print(model)
    # save the model
    if os.path.exists(options.model_saving_dir) is False:
        os.makedirs(options.model_saving_dir)
        with open(os.path.join(options.model_saving_dir, 'model.pkl'), 'wb') as f:
            parameters = options
            pickle.dump((parameters, model,proj_head), f)
        with open(os.path.join(options.model_saving_dir, 'res.txt'), 'w') as f:
            pass


def load_model(device,options):
    model_dir = options.model_saving_dir
    if os.path.exists(os.path.join(model_dir, 'model.pkl')) is False:
        return None,None,None

    # load the model and hyper-parameters (and maybe change some hyper-parameters if needed)
    with open(os.path.join(model_dir,'model.pkl'), 'rb') as f:
        param, classifier,proj_head = pickle.load(f)
        param.model_saving_dir = options.model_saving_dir
        classifier = classifier.to(device)
        proj_head = proj_head.to(device)
        if options.change_lr:
            param.learning_rate = options.learning_rate
        if options.change_alpha:
            param.alpha = options.alpha
    return param,classifier,proj_head

def unlabel_low(g, unlabel_threshold):
    mask_low = g.ndata['position'] <= unlabel_threshold
    g.ndata['label_o'][mask_low] = 0


# calculate the NCE loss
def NCEloss(pos1, pos2, neg, tao):
    pos_similarity = th.cosine_similarity(pos1, pos2, dim=-1)
    neg_similarity = th.cosine_similarity(pos1, neg, dim=-1)
    loss = -1 * th.log(
        th.exp(pos_similarity / tao)
        /
        (th.sum(th.exp(neg_similarity / tao)) - math.exp(1 / tao))
    )
    return loss

# shuffle the nids
def shuffle_nids(nids):
    res_nids = []
    nids1 ,nids2 = [],[]
    for i,nid in enumerate(nids):
        if i%2==0:
            nids1.append(nid)
        else:
            nids2.append(nid)
    randnum= random.randint(1,100)
    random.seed(randnum)
    random.shuffle(nids1)
    random.seed(randnum)
    random.shuffle(nids2)
    for i in range(len(nids1)):
        res_nids.append(nids1[i])
        res_nids.append(nids2[i])
    return res_nids

def train(options):
    batch_sizes = {}

    start_input, start_aug = options.start[0], options.start[1]
    end_input, end_aug = options.end[0], options.end[1]
    loss_thred = options.loss_thred
    th.multiprocessing.set_sharing_strategy('file_system')
    device = th.device("cuda:" + str(options.gpu) if th.cuda.is_available() else "cpu")

    # show the curriculum learning enviroment: listing dataset used to train the model
    data_path = options.datapath
    train_data_files = []
    cur_input =start_input
    while cur_input<= end_input:
        if cur_input==start_input:
            for aug_id in range(start_aug,4):
                train_data_files.append((cur_input,aug_id))
        elif cur_input==end_input:
            for aug_id in range(1, end_aug+1):
                train_data_files.append((cur_input, aug_id))
        else:
            for aug_id in range(1, 4):
                train_data_files.append((cur_input, aug_id))
        cur_input +=1

    for num_input,num_aug in train_data_files:
        if num_input == 3:
            batch_sizes[(num_input, num_aug)] = 16
        elif num_input == 5:
            batch_sizes[(num_input,num_aug)] = 350
        else:
            batch_sizes[(num_input,num_aug)] = 512

    print('The Curriculum Learning Environment is set as: ',train_data_files)


    num_epoch = options.num_epoch
    # do some preprocess
    if options.preprocess:
        preprocess(data_path, device, options)
        return
    print(options)

    # load the dataset
    print("----------------Loading data----------------")
    data_loaders = []

    # set the curriculum learning enviroment: from easy to hard
    # we first learn on simple netlists with less inputs and
    for i,(num_input,num_aug) in enumerate(train_data_files):
        file = os.path.join(data_path, 'i{}/aug{}.pkl'.format(num_input, num_aug))
        with open(file, 'rb') as f:
            train_g, POs, depth = pickle.load(f)
            train_g.ndata['f_input'] = th.ones(size=(train_g.number_of_nodes(), options.hidden_dim), dtype=th.float)
            train_g.ndata['temp'] = th.ones(size=(train_g.number_of_nodes(), options.hidden_dim), dtype=th.float)
            train_g.ndata['ntype2'] = th.argmax(train_g.ndata['ntype'], dim=1).squeeze(-1)

        data_size = len(POs)

        for po in POs:
            assert len(train_g.successors(po)) == 0
        if data_size > options.batch_size:
            data_size = int(len(POs) / options.batch_size) * options.batch_size
        POs = POs[:data_size]

        add_self_loop = False
        sampler = Sampler(depth * [options.degree], include_dst_in_src=False, add_self_loop=add_self_loop)
        print('aug{}, depth:{},num_nodes:{}, num_pos:{}'.format(num_aug, depth, train_g.number_of_nodes(), len(POs)))

        data_loaders.append(
            (num_input, num_aug, MyNodeDataLoader(
                False,
                train_g,
                POs,
                sampler,
                bs = batch_sizes[(num_input, num_aug)],
                batch_size=batch_sizes[(num_input, num_aug)],
                shuffle=False,
                drop_last=False,
            ))
        )


    print("Data successfully loaded")

    #load the model
    print('----------------Loadind the model----------------')
    options, model,proj_head = load_model(device, options)
    if model is None:
        print("No model, please prepocess first , or choose a pretrain model")
        return

    print("Model successfully loaded! ")
    print('\t', model)

    # set the optimizer
    optim = th.optim.Adam(
        model.parameters(), options.learning_rate, weight_decay=options.weight_decay
    )
    model.train()


    print("----------------Start training----------------")
    for num_input, aug_indx, data_loader in data_loaders:
        print('Currently used dataset: ({}, {})'.format(num_input,aug_indx))
        for epoch in range(num_epoch):
            POs = data_loader.nids
            g = data_loader.g
            POs = shuffle_nids(POs)
            # set the dataloader
            sampler = data_loader.block_sampler
            data_loader = MyNodeDataLoader(
                False,
                g,
                POs,
                sampler,
                bs=batch_sizes[(num_input, aug_indx)],
                batch_size=batch_sizes[(num_input, aug_indx)],
                shuffle=False,
                drop_last=False,
            )

            runtime = 0
            total_num, total_loss, correct, fn, fp, tn, tp = 0, 0.0, 0, 0, 0, 0, 0

            # each time we sample a batch of samples (netlists) and do the loss calculation
            # and back prooagation
            for ni, (central_nodes, input_nodes, blocks) in enumerate(data_loader):
                if ni==len(data_loader)-1:
                    continue
                start_time = time()
                # load the data to gpu
                blocks = [b.to(device) for b in blocks]
                loss = 0
                # generate the embedding for each sampled netlist
                embeddings = model(blocks, blocks[0].srcdata['f_input'])
                # calculate the NCE loss
                for i in range(0, len(embeddings), 2):
                    loss += NCEloss(embeddings[i], embeddings[i + 1], embeddings, options.tao)
                    loss += NCEloss(embeddings[i + 1], embeddings[i], embeddings, options.tao)
                loss = loss / len(embeddings)
                total_num += 1
                total_loss += loss
                endtime = time()
                runtime += endtime - start_time

                start_time = time()
                # backward propagation of the loss
                optim.zero_grad()
                loss.backward()
                optim.step()
                endtime = time()
                runtime += endtime - start_time

            # caculate the total loss of the current epoch
            Train_loss = total_loss / total_num

            print("epoch[{:d}]".format(epoch))
            print("training runtime: ", runtime)
            print("  train:")
            print("loss:{:.8f}".format(Train_loss.item()))

            # save the model
            judgement = True
            if judgement:
                print("Saving model.... ", os.path.join(options.model_saving_dir))
                if os.path.exists(options.model_saving_dir) is False:
                    os.makedirs(options.model_saving_dir)
                with open(os.path.join(options.model_saving_dir, 'model.pkl'), 'wb') as f:
                    parameters = options
                    pickle.dump((parameters, model,proj_head), f)
                print("Model successfully saved")
            if Train_loss.item() < loss_thred:
                print('train loss beyond thredshold, change to the next dataset: {} {}'.format(num_input, aug_indx))
                break


if __name__ == "__main__":
    seed = 1234
    # th.set_deterministic(True)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    np.random.seed(seed)
    train(get_options())