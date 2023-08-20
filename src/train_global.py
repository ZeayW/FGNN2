from dataset_gcl import *
from options import get_options
# from model import *
from FunctionConv import FuncConv,MLP
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
from torch.utils.data import DataLoader,Dataset
from dgl.dataloading import GraphDataLoader
from torch.nn.functional import softmax
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import datetime
from model import Classifier

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

options = get_options()
device = th.device("cuda:" + str(options.gpu) if th.cuda.is_available() else "cpu")

if options.nlabels != 1:
    Loss = nn.CrossEntropyLoss()
else:
    Loss = nn.BCEWithLogitsLoss(pos_weight=th.FloatTensor([options.pos_weight]).to(device))

label2block = {
    0: "adder",
    1: "multiplier",
    2: "divider",
    3: "subtractor"
}
def load_data(options):

    # load the dataset
    print("----------------Loading data----------------")
    num_train = {
        'adder':450,
        'multiplier': 550,
        'divider': 250,
        'subtractor': 250
    }
    num_val = {
        'adder': 100,
        'multiplier': 150,
        'divider': 50,
        'subtractor': 50
    }
    num_test = {
        'adder': 300,
        'multiplier': 500,
        'divider': 200,
        'subtractor': 150
    }
    total_num_train, total_num_test = 1500, 1150
    data_train = []
    data_test = []
    data_val = []
    data_path = options.datapath
    for target in ['train','test']:
        ratio = options.ratio * (total_num_test / total_num_train) if target == 'train' else 1
        num_map = num_train if target=='train' else num_test
        #print(target)
        target_dir = os.path.join(data_path, target)
        for data_file in os.listdir(target_dir):
            if not data_file.endswith('pkl'):
                continue
            block = data_file.split('.')[0]

            data_file_path = os.path.join(target_dir, data_file)
            with open(data_file_path, 'rb') as f:
                data = pickle.load(f)
                shuffle(data)
            new_data = []
            for graph, topo, PO_nids, label in data:
                topo = [t.to(device) for t in topo]
                PO_nids = th.tensor(PO_nids).to(device)
                graph.ndata['h'] = th.ones((graph.number_of_nodes(), options.hidden_dim), dtype=th.float)
                graph = graph.to(device)
                new_data.append((graph, topo, PO_nids, label))
            # print(len(data))
            if target == 'train':
                data_train.extend(
                    new_data[:int(ratio*num_map[block])]
                )
            elif target == 'test':
                data_train.extend(
                    new_data[:int(ratio * num_map[block])]
                )
                data_val.extend(
                    new_data[int(ratio * num_map[block]):]
                )
            #print('\t #{}: {}'.format(block, len(new_data[:num_map[block]])))

    labels_train = [d[3] for d in data_train]
    labels_test = [d[3] for d in data_test]
    labels_val = [d[3] for d in data_val]

    print(len(data_train),len(data_test),len(data_val))

    return data_train, labels_train, data_test, labels_test, data_val, labels_val



def init_model(options):
    eoncoder = FuncConv(
            hidden_dim=options.hidden_dim,
            out_dim = options.out_dim
        )
    if options.pre_train:
        model_save_path = '../checkpoints/{}'.format(options.start_point)
        assert os.path.exists(model_save_path), 'start_point {} does not exist'. \
            format(options.start_point)
        print('load a pretrained model from {}'.format(model_save_path))
        eoncoder.load_state_dict(th.load(model_save_path, map_location={'cuda:1': 'cuda:0'}))
    readout = MLP(options.out_dim*2,options.out_dim,int(options.out_dim/2),options.nlabels)
    model = Classifier(eoncoder, readout)
    print("creating model:")
    print(model)

    return model



def init(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def test(model,test_data,test_labels):
    with th.no_grad():
        embeddings = None
        count = options.nlabels*[0]
        correct_num = options.nlabels*[0]
        for graph, topo, PO_nids, label in test_data:
            # topo = [t.to(device) for t in topo]
            # PO_nids = th.tensor(PO_nids).to(device)
            # graph.ndata['h'] = th.ones((graph.number_of_nodes(), options.hidden_dim), dtype=th.float)
            # graph = graph.to(device)
            # print(label)
            count[label] += 1
            embedding = model.encoder(graph, topo, PO_nids)
            mean_embedding = th.mean(embedding, dim=0)
            max_embedding = th.max(embedding, dim=0).values
            global_embedding = th.cat((mean_embedding, max_embedding), dim=0)
            # print(embedding.shape,mean_embedding.shape)
            if embeddings is None:
                embeddings = global_embedding.unsqueeze(0)
            else:
                embeddings = th.cat((embeddings, global_embedding.unsqueeze(0)), dim=0)
            # print(embeddings.shape)
        labels = th.tensor(test_labels).to(device)
        labels_hat = model.readout(embeddings)
        predict_labels = th.argmax(softmax(labels_hat, 1), dim=1)
        test_loss = Loss(labels_hat, labels)
        total_correct = (
                predict_labels == labels
        ).sum().item()
        total_test_acc = total_correct / len(test_data)
        for j in range(options.nlabels):
            correct_num[j] += (predict_labels[labels == j] == j).sum().item()
        test_acc = [correct_num[j]/count[j] for j in range(options.nlabels)]

        # print("\ttp:", tp, " fp:", fp, " fn:", fn, " tn:", tn, " precision:", round(Train_precision,3))
        print("\t\tloss:{:.8f}, acc:{:.3f}, ".format(test_loss, total_test_acc))
        text = ""
        for j in range(options.nlabels):
            text += "{}: {:.3f},\t".format(label2block[j], test_acc[j])
        print("\t\t", text)

def train(model):
    print(options)
    loss_thred = options.loss_thred
    th.multiprocessing.set_sharing_strategy('file_system')

    data_train, labels_train, data_test, labels_test, data_val, labels_val = load_data(options)
    #print(list(range(len(data_train)))[:10],labels_train[:10])
    #print(zip(list(range(len(data_train))),labels_train)[:10])
    train_loader = DataLoader(list(zip(list(range(len(data_train))),labels_train)), batch_size=options.batch_size, shuffle=True, drop_last=True)
    print("Data successfully loaded")

    # set the optimizer
    optim = th.optim.Adam(
        model.parameters(), options.learning_rate, weight_decay=options.weight_decay
    )
    model.train()

    beta = options.beta


    print("----------------Start training----------------")
    cur_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    for epoch in range(options.num_epoch):
        total_num,total_loss, total_correct = 0,0.0,0
        count = options.nlabels*[0]
        correct_num = options.nlabels*[0]
        for i, (sample_indexs, labels) in enumerate(train_loader):
            #print(labels)
            embeddings = None
            for idx in sample_indexs:
                graph, topo, PO_nids, label = data_train[idx]
                # topo = [t.to(device) for t in topo]
                # PO_nids = th.tensor(PO_nids).to(device)
                # graph.ndata['h'] = th.ones((graph.number_of_nodes(), options.hidden_dim), dtype=th.float)
                # graph = graph.to(device)
                #print(label)
                count[label] += 1
                embedding = model.encoder(graph,topo,PO_nids)
                mean_embedding = th.mean(embedding, dim=0)
                max_embedding = th.max(embedding, dim=0).values
                global_embedding = th.cat((mean_embedding, max_embedding), dim=0)
                # print(embedding.shape,mean_embedding.shape)
                if embeddings is None:
                    embeddings = global_embedding.unsqueeze(0)
                else:
                    embeddings = th.cat((embeddings, global_embedding.unsqueeze(0)), dim=0)
            #print(embeddings.shape)
            total_num += len(labels)
            labels = labels.to(device)
            labels_hat = model.readout(embeddings)
            #print(labels_hat,labels_hat.shape)
            #print(labels,labels.shape)
            predict_labels = th.argmax(softmax(labels_hat, 1), dim=1)
            train_loss = Loss(labels_hat, labels)
            #print('loss:', train_loss.item())

            total_loss += train_loss.item() * len(labels)
            total_correct += (
                    predict_labels == labels
            ).sum().item()
            for j in range(options.nlabels):
                correct_num[j] += (predict_labels[labels==j] == j).sum().item()
            optim.zero_grad()
            train_loss.backward()
            # print(model.GCN1.layers[0].attn_n.grad)
            optim.step()
        Train_loss = total_loss / total_num
        total_train_acc = total_correct / total_num
        Train_acc = [correct_num[j]/count[j] for j in range(options.nlabels)]
        print("epoch[{:d}]".format(epoch))
        print(" \ttrain:")
        # print("\ttp:", tp, " fp:", fp, " fn:", fn, " tn:", tn, " precision:", round(Train_precision,3))
        print("\t\tloss:{:.8f}, acc:{:.3f}, ".format(Train_loss, total_train_acc))
        text = ""
        for j in range(options.nlabels):
            text += "{}: {:.3f},\t".format(label2block[j],Train_acc[j])
        print("\t\t",text)
        print(" \tval:")
        test(model, data_val, labels_val)
        print(" \ttest:")
        test(model,data_test,labels_test)
        if options.checkpoint:
            save_path = '../checkpoints/{}'.format(options.checkpoint)
            th.save(model.state_dict(), os.path.join(save_path,"{}.pth".format(epoch)))
            print('saved model to', os.path.join(save_path,"{}.pth".format(epoch)))


if __name__ == "__main__":
    seed = random.randint(1, 10000)
    seed = 9294
    init(seed)
    if options.test_iter:
        assert options.checkpoint, 'no checkpoint dir specified'
        model_save_path = '../checkpoints/{}/{}.pth'.format(options.checkpoint, options.test_iter)
        assert os.path.exists(model_save_path), 'start_point {} of checkpoint {} does not exist'.\
            format(options.test_iter, options.checkpoint)
        beta = options.beta
        options = th.load('../checkpoints/{}/options.pkl'.format(options.checkpoint))
        options.pre_train = False
        model = init_model(options)
        model = model.to(device)
        #model.load_state_dict(th.load(model_save_path, map_location=th.device('cpu')))
        model.load_state_dict(th.load(model_save_path,map_location={'cuda:1':'cuda:0'}))
        data_train, labels_train, data_test, labels_test, data_val, labels_val =  load_data(options)
        test(model, data_test, labels_test)

        # Loss = nn.CrossEntropyLoss()
        # validate([testdataloader], 'adder_o', device, model,Loss, beta, options)


    elif options.checkpoint:
        print('saving logs and models to ../checkpoints/{}'.format(options.checkpoint))
        checkpoint_path = '../checkpoints/{}'.format(options.checkpoint)
        stdout_f = '../checkpoints/{}/stdout.log'.format(options.checkpoint)
        stderr_f = '../checkpoints/{}/stderr.log'.format(options.checkpoint)
        os.makedirs(checkpoint_path)  # exist not ok
        th.save(options, os.path.join(checkpoint_path, 'options.pkl'))
        with tee.StdoutTee(stdout_f), tee.StderrTee(stderr_f):
            model = init_model(options)
            model = model.to(device)
            print('seed:', seed)
            train(model)

    else:
        print('No checkpoint is specified. abandoning all model checkpoints and logs')
        model = init_model(options)
        model = model.to(device)
        train(model)