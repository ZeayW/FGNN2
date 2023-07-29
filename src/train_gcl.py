from dataset_gcl import *
from options import get_options
# from model import *
from FunctionConv import FuncConv
import dgl
import pickle
import numpy as np
import os
from time import time
import math
import networkx as nx
from random import shuffle
import random
import torch as th
from torch.utils.data import DataLoader
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import datetime

class MyLoader(th.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

options = get_options()
device = th.device("cuda:" + str(options.gpu) if th.cuda.is_available() else "cpu")


def load_data(options):
    batch_sizes = {}
    start_input, start_aug = options.start[0], options.start[1]
    end_input, end_aug = options.end[0], options.end[1]

    data_path = options.datapath
    traindata_series = []
    cur_input = start_input
    while cur_input <= end_input:
        s = start_aug if cur_input==start_input else 1
        e = end_aug+1 if cur_input==end_input else 4
        traindata_series.extend(
            [(cur_input, i ) for i in options.per2replace[s-1:e]]
        )
        cur_input += 1

    for num_input, num_aug in traindata_series:
        if num_input == 4:
            batch_sizes[(num_input, num_aug)] = 256
        elif num_input == 5:
            batch_sizes[(num_input, num_aug)] = 350
        else:
            batch_sizes[(num_input, num_aug)] = 512

    print('The Curriculum Learning Environment is set as: ', traindata_series)

    # load the dataset
    print("----------------Loading data----------------")
    data_loaders = []

    # set the curriculum learning enviroment: from easy to hard
    # we first learn on simple netlists with less inputs and
    for i, (num_input, num_aug) in enumerate(traindata_series):
        file = os.path.join(data_path, 'i{}/aug{}.pkl'.format(num_input, num_aug))
        assert os.path.exists(file), \
            'i{}/aug{}.pkl is missing, Please call dataset_gcl.py to generate dataset first!'.format(num_input, num_aug)

        with open(file, 'rb') as f:
            positive_pairs = pickle.load(f)
            # train_g.ndata['f_input'] = th.ones(size=(train_g.number_of_nodes(), options.hidden_dim), dtype=th.float)
        print(len(positive_pairs))
        sampler = SubsetRandomSampler(th.arange(len(positive_pairs)))
        loader = GraphDataLoader(MyLoader(positive_pairs), sampler=sampler,batch_size=batch_sizes[(num_input, num_aug)], drop_last=True)
        data_loaders.append(
            (num_input, num_aug, loader)
        )

    return data_loaders



def init_model(options):
    model = FuncConv(
            hidden_dim=options.hidden_dim,
            out_dim = options.out_dim
        )
    print("creating model:")
    print(model)

    return model

def unlabel_low(g, unlabel_threshold):
    mask_low = g.ndata['position'] <= unlabel_threshold
    g.ndata['label_o'][mask_low] = 0


def NCEloss_w2(embeddings, codes,i, j, tao):
    pos_similarity = th.cosine_similarity(embeddings[i], embeddings[j], dim=-1)
    neg_similarity_1 = th.cosine_similarity(embeddings[i], embeddings, dim=-1)
    neg_similarity_2 = th.cosine_similarity(embeddings[j], embeddings, dim=-1)
    code_similarity = th.cosine_similarity(codes[i], codes, dim=-1)
    #print(pos_similarity,neg_similarity_1,neg_similarity_2)

    loss_12 = -1 * th.log(
        th.exp(pos_similarity / tao)
                    /
        (th.sum(th.exp((1-code_similarity)*neg_similarity_1 / tao)) -1 )
    )
    loss_21 = -1 * th.log(
        th.exp(pos_similarity / tao)
        /
        ( th.sum(th.exp((1-code_similarity)*neg_similarity_2 / tao)) -1 )
    )
    return loss_12 + loss_21

def NCEloss_w(embeddings, codes,i, j, tao):
    pos_similarity = th.cosine_similarity(embeddings[i], embeddings[j], dim=-1)
    neg_similarity_1 = th.cosine_similarity(embeddings[i], embeddings, dim=-1)
    neg_similarity_2 = th.cosine_similarity(embeddings[j], embeddings, dim=-1)
    code_similarity = th.cosine_similarity(codes[i], codes, dim=-1)
    #print(pos_similarity,neg_similarity_1,neg_similarity_2)

    loss_12 = -1 * th.log(
        th.exp(pos_similarity / tao)
                    /
        (th.sum((1-code_similarity) * th.exp(neg_similarity_1 / tao)) +  th.exp(pos_similarity / tao) )
    )
    loss_21 = -1 * th.log(
        th.exp(pos_similarity / tao)
        /
        ( th.sum((1-code_similarity) *th.exp(neg_similarity_2 / tao)) +  th.exp(pos_similarity / tao) )
    )
    return loss_12 + loss_21

# calculate the NCE loss
def NCEloss(embeddings,codes, i, j, tao):
    pos_similarity = th.cosine_similarity(embeddings[i], embeddings[j], dim=-1)
    neg_similarity_1 = th.cosine_similarity(embeddings[i], embeddings, dim=-1)
    neg_similarity_2 = th.cosine_similarity(embeddings[j], embeddings, dim=-1)
    #print(pos_similarity,neg_similarity_1,neg_similarity_2)
    loss_12 = -1 * th.log(
        th.exp(pos_similarity / tao)
                    /
        (th.sum(th.exp(neg_similarity_1 / tao)) - math.exp(1 / tao))
    )
    loss_21 = -1 * th.log(
        th.exp(pos_similarity / tao)
        /
        (th.sum(th.exp(neg_similarity_2 / tao)) - math.exp(1 / tao))
    )
    return loss_12 + loss_21

# def NCEloss(pos1, pos2, neg, tao):
#     pos_similarity = th.cosine_similarity(pos1, pos2, dim=-1)
#     neg_similarity = th.cosine_similarity(pos1, neg, dim=-1)
#     loss = -1 * th.log(
#         th.exp(pos_similarity / tao)
#         /
#         (th.sum(th.exp(neg_similarity / tao)) - math.exp(1 / tao))
#     )
#     return loss

def init(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def train(model):
    print(options)
    loss_thred = options.loss_thred
    th.multiprocessing.set_sharing_strategy('file_system')
    data_loaders = load_data(options)
    print("Data successfully loaded")

    # set the optimizer
    optim = th.optim.Adam(
        model.parameters(), options.learning_rate, weight_decay=options.weight_decay
    )
    model.train()

    Loss = NCEloss_w if options.weighted else NCEloss
    print("----------------Start training----------------")
    cur_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    for num_input, aug_percent, data_loader in data_loaders:
        save_path = '../checkpoints/{}/i{}_aug{}'.format(options.checkpoint, num_input, aug_percent)
        os.makedirs(save_path,exist_ok=True)
        print('Currently used curriculum: ({}, {})'.format(num_input,aug_percent))
        for epoch in range(options.num_epoch):
            seed = random.randint(1, 10000)
            init(seed)
            total_num, total_loss = 0,0
            for i, samples_pair in enumerate(data_loader):
                loss = 0
                embeddings = [None,None]
                codes = None
                for j, (cd,graph, PO_nids, sizes) in enumerate(samples_pair):
                    codes = cd
                    graph.ndata['temp'] = th.ones(size=(graph.number_of_nodes(), options.hidden_dim),
                                                   dtype=th.float)
                    graph.ndata['h'] = th.ones((graph.number_of_nodes(), model.hidden_dim), dtype=th.float)
                    graph = graph.to(device)
                    PO_nids = PO_nids.to(device)
                    topo_levels = dgl.topological_nodes_generator(graph)
                    topo_levels = [l.to(device) for l in topo_levels]
                    shift = 0
                    for k, size in enumerate(sizes):
                        PO_nids[k] += shift
                        shift += size
                    # print(topo,PO_nids)
                    # print(graph.ndata['output'][PO_nids])
                    # print(graph.ndata['v'][PO_nids])
                    embeddings[j] = model(graph,topo_levels,PO_nids)
                    # if i == 0:
                    #     print(code[:10])
                    #     print(embeddings[j][:10])
                    #print(embeddings[j])
                codes = th.cat((codes,codes)).to(device)
                num_pair = len(embeddings[0])
                embeddings = th.cat((embeddings[0],embeddings[1]))
                # calculate the NCE loss
                for j in range(num_pair):
                    loss += Loss(embeddings,codes,j,j+num_pair,options.tao)
                loss = loss / (2*num_pair)
                total_num += 1
                total_loss += loss
                # if i%10==0: print("loss: ",loss.item())
                # backward propagation of the loss
                optim.zero_grad()
                loss.backward()
                optim.step()

            total_loss = total_loss / total_num
            print("epoch: {}, loss: {}".format(epoch, total_loss.item()))
            if options.checkpoint:
                #save_path = '../checkpoints/{}/i{}_aug{}/{}.pth'.format(options.checkpoint, num_input, aug_percent, epoch)
                th.save(model.state_dict(), os.path.join(save_path,"{}.pth".format(epoch)))
                print('saved model to', os.path.join(save_path,"{}.pth".format(epoch)))
            if total_loss.item() < loss_thred:
                print('train loss beyond thredshold, change to the next curriculum setting')
                break


if __name__ == "__main__":
    # seed = random.randint(1, 10000)
    # seed = 9294
    # init(seed)

    model = init_model(options).to(device)
    if options.start_point:
        pretrained_path = '../checkpoints/{}'.format(options.start_point)
        assert os.path.exists(pretrained_path), 'start_point {} does not exist'.\
            format(options.start_point)
        #options = th.load('../checkpoints/{}/options.pkl'.format(options.checkpoint))
        model.load_state_dict(th.load(pretrained_path))
        print("continue training from checkpoint {}".format(options.start_point))

    if options.checkpoint:
        checkpoint_path = '../checkpoints/{}'.format(options.checkpoint)
        os.makedirs(checkpoint_path)  # exist not ok
        th.save(options, os.path.join(checkpoint_path, 'options.pkl'))
        # os.makedirs('../checkpoints/{}'.format(options.checkpoint))  # exist not ok
        stdout_f = '../checkpoints/{}/stdout.log'.format(options.checkpoint)
        stderr_f = '../checkpoints/{}/stderr.log'.format(options.checkpoint)
        with tee.StdoutTee(stdout_f), tee.StderrTee(stderr_f):
            #print('seed:',seed)
            train(model)
    else:
        print('No checkpoint is specified. abandoning all model checkpoints and logs')
        train(model)