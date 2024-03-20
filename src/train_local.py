from dataset_gcl import *
from options import get_options
from model import *
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
from dgl.dataloading import GraphDataLoader,NodeDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

def type_count(ntypes,count):
    for tp in ntypes:
        tp = tp.item()
        count[tp] +=1

def cal_ratios(count1,count2):
    ratios = []
    for i in range(len(count1)):
        if count2[i] == 0:
            ratios.append(-1)
        else:
            ratio = count1[i] / count2[i]
            ratios.append(round(ratio,4))
    return ratios

def oversample(g,options,in_dim):
    r"""

    oversample the postive nodes when the dataset is imbalanced

    :param g:
        the target graph
    :param options:
        some args
    :param in_dim:
        number of different node types
    :return:
    """
    print("oversampling dataset......")

    print("total number of nodes: ", g.num_nodes())


    labels = g.ndata['adder_o']
    lowbit_mask = g.ndata['position']<=3
    # unlabel the nodes in muldiv
    no_muldiv_mask = labels.squeeze(-1)!=-1
    print('no_mul',len(labels[no_muldiv_mask]))
    nodes = th.tensor(range(g.num_nodes()))
    nodes = nodes[no_muldiv_mask]
    labels = labels[no_muldiv_mask]
    print(len(nodes))

    mask_pos = (labels ==1).squeeze(1)

    mask_neg = (labels == 0).squeeze(1)
    pos_nodes = nodes[mask_pos].numpy().tolist()
    neg_nodes = nodes[mask_neg].numpy().tolist()
    shuffle(pos_nodes)
    shuffle(neg_nodes)
    pos_size = len(pos_nodes)
    neg_size = len(neg_nodes)

    ratio = float(neg_size) / float(pos_size)
    print("ratio=", ratio)


    pos_count = th.zeros(size=(1, in_dim)).squeeze(0).numpy().tolist()
    neg_count = th.zeros(size=(1, in_dim)).squeeze(0).numpy().tolist()
    pos_types = g.ndata['ntype'][pos_nodes]
    neg_types = g.ndata['ntype'][neg_nodes]
    pos_types = th.argmax(pos_types, dim=1)
    neg_types = th.argmax(neg_types, dim=1)
    type_count(pos_types, pos_count)
    type_count(neg_types, neg_count)

    print("train pos count:", pos_count)
    print("train neg count:", neg_count)
    rates = cal_ratios(neg_count, pos_count)
    print(rates)

    train_nodes = pos_nodes.copy()
    train_nodes.extend(neg_nodes)

    ratios = []
    for type in range(in_dim):
        pos_mask = pos_types == type
        neg_mask = neg_types == type
        pos_nodes_n = th.tensor(pos_nodes)[pos_mask].numpy().tolist()
        neg_nodes_n = th.tensor(neg_nodes)[neg_mask].numpy().tolist()

        if len(pos_nodes_n) == 0: ratio = 0
        else: ratio = len(neg_nodes_n) / len(pos_nodes_n)
        ratios.append(ratio)
        if ratio >options.os_rate : ratio = options.os_rate

        if options.balanced and ratio!=0:
            if ratio > 1:
                short_nodes = pos_nodes_n
            else:
                short_nodes = neg_nodes_n
                ratio = 1 / ratio
            short_len = len(short_nodes)
            while ratio > 1:
                shuffle(short_nodes)
                train_nodes.extend(short_nodes[:int(short_len * min(1, ratio - 1))])
                ratio -= 1

    print("ratios:",ratios)
    return train_nodes,pos_count, neg_count



class MyLoader(th.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

def load_data(options):
    batch_sizes = {}
    start_input, start_aug = options.start[0], options.start[1]
    end_input, end_aug = options.end[0], options.end[1]

    data_path = options.datapath
    train_data_file = os.path.join(data_path, 'BOOM.pkl')
    val_data_file = os.path.join(data_path, 'RocketCore.pkl')

    with open(train_data_file,'rb') as f:
        train_g,_ = pickle.load(f)
        train_graphs = dgl.unbatch(train_g)
        if options.train_percent == 1:
            train_graphs = [train_graphs[3]]
        else:
            train_graphs = train_graphs[:int(options.train_percent)]
        train_g = dgl.batch(train_graphs)
        train_g_topo = dgl.topological_nodes_generator(train_g)
        train_g.ndata['adder_o'][train_g.ndata['mul_o']==1] = 0
        train_g.ndata['adder_o'][train_g.ndata['position']<=3] = 0
        train_g.ndata['ntype2'] = th.argmax(train_g.ndata['ntype'], dim=1).squeeze(-1)
    with open(val_data_file,'rb') as f:
        val_g,_ = pickle.load(f)
        val_g.ndata['adder_o'][val_g.ndata['mul_o'] == 1] = 0
        val_g.ndata['adder_o'][val_g.ndata['position'] <= 3] = 0
        val_g_topo = dgl.topological_nodes_generator(val_g)
        val_g.ndata['ntype2'] = th.argmax(val_g.ndata['ntype'], dim=1).squeeze(-1)
    return train_g, train_g_topo, val_g,val_g_topo


options = get_options()
device = th.device("cuda:" + str(options.gpu) if th.cuda.is_available() else "cpu")


def init_model(options):
    model = FuncConv(
            ntypes=options.ntypes,
            hidden_dim=options.hidden_dim,
            out_dim = options.out_dim
        )
    classifier = MLP(
        in_dim=options.out_dim,
        out_dim=options.nlabels,
        nlayers=options.n_fcn,
        dropout=options.mlp_dropout
    )

    print("creating model:")
    print(model)
    print(classifier)

    return model,classifier

def unlabel_low(g, unlabel_threshold):
    mask_low = g.ndata['position'] <= unlabel_threshold
    g.ndata['label_o'][mask_low] = 0


# calculate the NCE loss
def NCEloss(embeddings, i, j, tao):
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


def val(val_g,val_g_topo,model,classifier,Loss,beta):
    model.eval()
    classifier.eval()
    total_num, total_loss, correct, fn, fp, tn, tp = 0, 0.0, 0, 0, 0, 0, 0

    with th.no_grad():
        val_g.ndata['temp'] = th.ones(size=(val_g.number_of_nodes(), options.hidden_dim),
                                        dtype=th.float)
        val_g.ndata['h'] = th.ones((val_g.number_of_nodes(), model.hidden_dim), dtype=th.float)
        val_g = val_g.to(device)
        labels_gd = val_g.ndata['adder_o'].squeeze(1)
        total_num += len(labels_gd)
        topo_levels = [l.to(device) for l in val_g_topo]
        embeddings = model(val_g, topo_levels, None)
        labels_hat = classifier(embeddings).squeeze(0)
        if get_options().nlabels != 1:
            pos_prob = nn.functional.softmax(labels_hat, 1)[:, 1]
        else:
            pos_prob = th.sigmoid(labels_hat)
        pos_prob[pos_prob >= beta] = 1
        pos_prob[pos_prob < beta] = 0
        predict_labels = pos_prob
        #val_loss = Loss(labels_hat, labels_gd)
        #total_loss += val_loss.item() * len(labels_gd)

        correct += (
                predict_labels == labels_gd
        ).sum().item()

        # count fake negatives (fn), true negatives (tp), true negatives (tn), true post
        fn += ((predict_labels == 0) & (labels_gd != 0)).sum().item()
        tp += ((predict_labels != 0) & (labels_gd != 0)).sum().item()
        tn += ((predict_labels == 0) & (labels_gd == 0)).sum().item()
        fp += ((predict_labels != 0) & (labels_gd == 0)).sum().item()

        Val_loss = total_loss / total_num

        # calculate accuracy, recall, precision and F1-score
        Val_acc = correct / total_num
        Val_recall = 0
        Val_precision = 0
        if tp != 0:
            Val_recall = tp / (tp + fn)
            Val_precision = tp / (tp + fp)
        Val_F1_score = 0
        if Val_precision != 0 or Val_recall != 0:
            Val_F1_score = 2 * Val_recall * Val_precision / (Val_recall + Val_precision)

        print("  val:")
        print("\ttp:", tp, " fp:", fp, " fn:", fn, " tn:", tn, " precision:", round(Val_precision, 3))
        print("\tloss:{:.8f}, acc:{:.3f}, recall:{:.3f}, F1 score:{:.3f}".format(Val_loss, Val_acc, Val_recall,
                                                                                 Val_F1_score))


def train(model,classifier):
    print(options)
    loss_thred = options.loss_thred
    th.multiprocessing.set_sharing_strategy('file_system')
    print("Data successfully loaded")

    train_g, train_g_topo, val_g, val_g_topo = load_data(options)
    train_nodes, pos_count, neg_count = oversample(train_g, options, options.ntypes)
    print(len(train_nodes))
    # set the optimizer
    Loss = nn.CrossEntropyLoss()
    optim = th.optim.Adam([
        {'params':model.parameters(),'lr':options.learning_rate},
        {'params': classifier.parameters(), 'lr': options.learning_rate}
    ])
    # optim = th.optim.Adam(
    #     model.parameters(), options.learning_rate, weight_decay=options.weight_decay
    # )
    model.train()
    classifier.train()

    beta = options.beta
    pre_loss = 100
    print("----------------Start training----------------")
    # train_g.ndata['temp'] = th.ones(size=(train_g.number_of_nodes(), options.hidden_dim),
    #                                 dtype=th.float)
    # train_g.ndata['h'] = th.ones((train_g.number_of_nodes(), model.hidden_dim))
    train_graphs = [(train_g, train_g_topo)]
    for epoch in range(options.num_epoch):
        total_num, total_loss, correct, fn, fp, tn, tp = 0, 0.0, 0, 0, 0, 0, 0
        for graph,topo in train_graphs:
            sampler = SubsetRandomSampler(th.tensor(train_nodes))
            loader = DataLoader(MyLoader(range(graph.number_of_nodes())), sampler=sampler,
                                batch_size=options.batch_size, drop_last=True)
            graph = graph.to(device)

            for j, nodes in enumerate(loader):
                graph.ndata['temp'] = th.ones(size=(graph.number_of_nodes(), options.hidden_dim),
                                                dtype=th.float).to(device)
                graph.ndata['h'] = th.ones((graph.number_of_nodes(), model.hidden_dim)).to(device)
                #print(nodes)
                #print(train_g_topo)
                #print(len(nodes))
                #print(len(train_g_topo),train_g.number_of_edges())
                th.cuda.empty_cache()
                # train_g.ndata['temp'] = th.ones(size=(train_g.number_of_nodes(), options.hidden_dim),
                #                               dtype=th.float).to(device)
                # train_g.ndata['h'] = th.ones((train_g.number_of_nodes(), model.hidden_dim), dtype=th.float).to(device)
                labels_gd = graph.ndata['adder_o'][nodes].squeeze(1)
                total_num += len(labels_gd)
                topo_levels = [l.to(device) for l in topo]
                embeddings = model(graph, topo_levels, None).squeeze(0)[nodes]
                #print(embeddings,len(embeddings))
                labels_hat = classifier(embeddings)
                if get_options().nlabels != 1:
                    pos_prob = nn.functional.softmax(labels_hat, 1)[:, 1]
                else:
                    pos_prob = th.sigmoid(labels_hat)
                pos_prob[pos_prob >= beta] = 1
                pos_prob[pos_prob < beta] = 0
                predict_labels = pos_prob
                #print(labels_hat.shape,labels_gd.shape)
                train_loss = Loss(labels_hat, labels_gd)
                print(train_loss)
                total_loss += train_loss.item() * len(labels_gd)

                correct += (
                        predict_labels == labels_gd
                ).sum().item()

                # count fake negatives (fn), true negatives (tp), true negatives (tn), true post
                fn += ((predict_labels == 0) & (labels_gd != 0)).sum().item()
                tp += ((predict_labels != 0) & (labels_gd != 0)).sum().item()
                tn += ((predict_labels == 0) & (labels_gd == 0)).sum().item()
                fp += ((predict_labels != 0) & (labels_gd == 0)).sum().item()

                optim.zero_grad()
                train_loss.backward()
                optim.step()
                th.cuda.empty_cache()
                if j%10==0:
                    val(val_g, val_g_topo, model, classifier, Loss, beta)

        Train_loss = total_loss / total_num

        if Train_loss > pre_loss:
            stop_score += 1
            if stop_score >= 2:
                print('Early Stop!')
                # exit()
        else:
            stop_score = 0
            pre_loss = Train_loss

        # calculate accuracy, recall, precision and F1-score
        Train_acc = correct / total_num
        Train_recall = 0
        Train_precision = 0
        if tp != 0:
            Train_recall = tp / (tp + fn)
            Train_precision = tp / (tp + fp)
        Train_F1_score = 0
        if Train_precision != 0 or Train_recall != 0:
            Train_F1_score = 2 * Train_recall * Train_precision / (Train_recall + Train_precision)

        print("epoch[{:d}]".format(epoch))
        #print("training runtime: ", runtime)
        print("  train:")
        print("\ttp:", tp, " fp:", fp, " fn:", fn, " tn:", tn, " precision:", round(Train_precision, 3))
        print("\tloss:{:.8f}, acc:{:.3f}, recall:{:.3f}, F1 score:{:.3f}".format(Train_loss, Train_acc, Train_recall,
                                                                                 Train_F1_score))
        val(val_g,val_g_topo,model, classifier, Loss, beta)

def init(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    seed = random.randint(1, 10000)
    seed = 9294
    init(seed)
    if options.start_iter:
        assert options.checkpoint, 'no checkpoint dir specified'
        model_save_path = '../checkpoints/{}/{}.pth'.format(options.checkpoint, options.start_iter)
        assert os.path.exists(model_save_path), 'start_iter {} of checkpoint {} does not exist'.\
            format(options.start_iter, options.checkpoint)
        options = th.load('../checkpoints/{}/options.pkl'.format(options.checkpoint))
        model, classifier = init_model(options)
        model = model.to(device)
        classifier = classifier.to(device)
        model.load_state_dict(th.load(model_save_path))
        stdout_f = '../checkpoints/{}/stdout_{}.log'.format(options.checkpoint,options.start_iter)
        stderr_f = '../checkpoints/{}/stderr_{}.log'.format(options.checkpoint,options.start_iter)
        with tee.StdoutTee(stdout_f), tee.StderrTee(stderr_f):
            print("continue training from {}".format(options.start_iter))
            print('seed:',seed)
            train(model, classifier)

    elif options.checkpoint:
        print('saving logs and models to ../checkpoints/{}'.format(options.checkpoint))
        checkpoint_path = '../checkpoints/{}'.format(options.checkpoint)
        os.makedirs(checkpoint_path)  # exist not ok
        th.save(options, os.path.join(checkpoint_path, 'options.pkl'))
        model, classifier = init_model(options)
        model = model.to(device)
        classifier = classifier.to(device)
        # os.makedirs('../checkpoints/{}'.format(options.checkpoint))  # exist not ok
        stdout_f = '../checkpoints/{}/stdout.log'.format(options.checkpoint)
        stderr_f = '../checkpoints/{}/stderr.log'.format(options.checkpoint)
        with tee.StdoutTee(stdout_f), tee.StderrTee(stderr_f):
            print('seed:',seed)
            train(model, classifier)
    else:
        print('No checkpoint is specified. abandoning all model checkpoints and logs')
        model, classifier = init_model(options)
        model = model.to(device)
        classifier = classifier.to(device)
        train(model, classifier)