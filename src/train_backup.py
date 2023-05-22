r"""
this script is used to train/fine-tune and validate/test the models
"""

from dataset import *
from options import get_options
from model import *
import dgl
import pickle
import numpy as np
import os
from MyDataLoader import *
from time import time
from random import shuffle
import itertools


def DAG2UDG(g):
    r"""

    used to transform a (directed acyclic graph)DAG into a undirected graph

    :param g: dglGraph
        the target DAG

    :return:
        dglGraph
            the output undirected graph
    """
    edges = g.edges()
    reverse_edges = (edges[1],edges[0])
    # add the reversed edges
    new_edges = (th.cat((edges[0],reverse_edges[0])),th.cat((edges[1],reverse_edges[1])))
    udg =  dgl.graph(new_edges,num_nodes=g.num_nodes())

    # copy the node features
    for key, value in g.ndata.items():
        # print(key,value)
        udg.ndata[key] = value
    # copy the edge features
    udg.edata['direction'] = th.cat((th.zeros(size=(1,g.number_of_edges())).squeeze(0),th.ones(size=(1,g.number_of_edges())).squeeze(0)))

    return udg

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



    if options.label == 'in':
        labels = g.ndata['label_i']
    elif options.label == 'out':
        labels = g.ndata['label_o']

    else:
        print("wrong label type")
        return
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


def preprocess(data_path,device,options):
    r"""

    do some preprocessing work: generate dataset / initialize the model

    :param data_path:
        the path to save the dataset
    :param device:
        the device to load the model
    :param options:
        some additional parameters
    :return:
        no return
    """

    label2id = {}
    if os.path.exists(data_path) is False:
        os.makedirs(data_path)
    train_data_file = os.path.join(data_path, 'boom.pkl')
    val_data_file = os.path.join(data_path, 'rocket.pkl')

    # generate and save the test dataset if missing
    if os.path.exists(val_data_file) is False:
        datapaths = ["../dc/rocket/implementation/"]
        report_folders = ["../dc/rocket/report/"]
        th.multiprocessing.set_sharing_strategy('file_system')
        dataset = Dataset("Rocket",datapaths,report_folders,label2id)
        g = dataset.batch_graph
        with open(val_data_file,'wb') as f:
            pickle.dump(g,f)

    # generate and save the train dataset if missing
    if os.path.exists(train_data_file) is False:
        datapaths = ["../dc/boom/implementation/"]
        report_folders = ["../dc/boom/report/"]
        th.multiprocessing.set_sharing_strategy('file_system')
        dataset = Dataset("BoomCore", datapaths, report_folders, label2id)
        g = dataset.batch_graph
        with open(train_data_file,'wb') as f:
            pickle.dump(g,f)

    if len(label2id) != 0:
        with open(os.path.join(data_path,'label2id.pkl'),'wb') as f:
            pickle.dump(label2id,f)

    # initialize the  model
    if options.gat:
        network = GAT
    elif options.gin:
        network = GIN
    elif options.function:
        network = FuncGNN
    else:
        print('please choose a valid model type!')
        exit()

    model = network(
        ntypes = options.in_dim,
        hidden_dim=options.hidden_dim,
        out_dim=options.out_dim,
        dropout=options.gcn_dropout,
    )

    # initialize a multlayer perceptron
    mlp = MLP(
        in_dim = model.out_dim,
        out_dim = options.nlabels,
        nlayers = options.n_fcn,
        dropout = options.mlp_dropout
    ).to(device)
    print(model)
    print(mlp)
    print("creating model in:",options.model_saving_dir)
    # save the model and create a file a save the results
    if os.path.exists(options.model_saving_dir) is False:
        os.makedirs(options.model_saving_dir)
        with open(os.path.join(options.model_saving_dir, 'model.pkl'), 'wb') as f:
            parameters = options
            pickle.dump((parameters, model,mlp), f)
        with open(os.path.join(options.model_saving_dir, 'res.txt'), 'w') as f:
            pass


def load_model(device,options):
    r"""
    Load the model

    :param device:
        the target device that the model is loaded on
    :param options:
        some additional parameters
    :return:
        param: new options
        model : loaded model
        mlp: loaded mlp
    """
    model_dir = options.model_saving_dir
    if os.path.exists(os.path.join(model_dir, 'model.pkl')) is False:
        return None,None


    with open(os.path.join(model_dir,'model.pkl'), 'rb') as f:
        param, model,mlp = pickle.load(f)
        param.model_saving_dir = options.model_saving_dir
        model = model.to(device)

        # make some changes to the options
        if options.change_lr:
            param.learning_rate = options.learning_rate
        if options.change_alpha:
            param.alpha = options.alpha
    return param,model,mlp



def validate(loaders,label_name,device,model,mlp,Loss,beta,options):
    r"""

    validate the model

    :param loaders:
        the loaders to load the validation dataset
    :param label_name:
        target label name
    :param device:
        device
    :param model:
        trained model
    :param mlp:
        trained mlp
    :param Loss:
        used loss function
    :param beta:
        a hyperparameter that determines the thredshold of binary classification
    :param options:
        some parameters
    :return:
        result of the validation: loss, acc,recall,precision,F1_score
    """

    total_num, total_loss, correct, fn, fp, tn, tp = 0, 0.0, 0, 0, 0, 0, 0
    runtime = 0

    with th.no_grad():
        for i,loader in enumerate(loaders):
            for ni, (central_nodes, input_nodes, blocks) in enumerate(loader):
                start = time()
                blocks = [b.to(device) for b in blocks]

                # get the input features
                if options.gnn:
                    input_features = blocks[0].srcdata["ntype"]
                else:
                    input_features = blocks[0].srcdata["f_input"]

                # the central nodes are the output of the final block
                output_labels = blocks[-1].dstdata[label_name].squeeze(1)
                total_num += len(output_labels)
                # get the embeddings of central nodes
                embedding = model(blocks, input_features)

                # feed the embeddings into the mlp to predict the labels
                label_hat = mlp(embedding)
                pos_prob = nn.functional.softmax(label_hat, 1)[:, 1]
                # adjust the predicted labels based on a given thredshold beta
                pos_prob[pos_prob >= beta] = 1
                pos_prob[pos_prob < beta] = 0
                predict_labels = pos_prob

                end = time()
                runtime += end - start

                # calculate the loss
                val_loss = Loss(label_hat, output_labels)
                total_loss += val_loss.item() * len(output_labels)

                correct += (
                        predict_labels == output_labels
                ).sum().item()

                # count fake negatives (fn), true negatives (tp), true negatives (tn), true postives (tp)
                fn += ((predict_labels == 0) & (output_labels != 0)).sum().item()
                tp += ((predict_labels != 0) & (output_labels != 0)).sum().item()
                tn += ((predict_labels == 0) & (output_labels == 0)).sum().item()
                fp += ((predict_labels != 0) & (output_labels == 0)).sum().item()

    loss = total_loss / total_num
    acc = correct / total_num

    # calculate recall, precision and F1-score
    recall = 0
    precision = 0
    if tp != 0:
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
    F1_score = 0
    if precision != 0 or recall != 0:
        F1_score = 2 * recall * precision / (recall + precision)

    print("\ttp:", tp, " fp:", fp, " fn:", fn, " tn:", tn, " precision:", round(precision, 3))
    print("\tloss:{:.3f}, acc:{:.3f}, recall:{:.3f}, F1 score:{:.3f}".format(loss, acc,recall, F1_score))

    return [loss, acc,recall,precision,F1_score]


def train(options):

    th.multiprocessing.set_sharing_strategy('file_system')
    device = th.device("cuda:"+str(options.gpu) if th.cuda.is_available() else "cpu")

    data_path = options.datapath
    print(data_path)
    train_data_file = os.path.join(data_path,'boom.pkl')
    val_data_file = os.path.join(data_path,'rocket.pkl')

    # preprocess: generate dataset / initialize the model
    if options.preprocess :
        preprocess(data_path,device,options)
        return
    print(options)
    # load the model
    options, model,mlp = load_model(device, options)
    if model is None:
        print("No model, please prepocess first , or choose a pretrain model")
        return
    print(model)
    mlp = mlp.to(device)
    print(mlp)

    in_nlayers = options.in_nlayers if isinstance(options.in_nlayers,int) else options.in_nlayers[0]
    out_nlayers = options.out_nlayers if isinstance(options.out_nlayers,int) else options.out_nlayers[0]

    label_name = 'label_o'
    print("Loading data...")
    with open(train_data_file,'rb') as f:
        train_g = pickle.load(f)
        train_graphs = dgl.unbatch(train_g)
        train_graphs = train_graphs[:int(options.train_percent)]
        train_g = dgl.batch(train_graphs)
    with open(val_data_file,'rb') as f:
        val_g = pickle.load(f)

    #val_g = val_g[:int(len())]
    print("num train pos", len(train_g.ndata['label_o'][train_g.ndata['label_o'].squeeze(1) >0]))
    print("num val pos", len(val_g.ndata['label_o'][val_g.ndata['label_o'].squeeze(1) >0]))


    train_nodes, pos_count, neg_count = oversample(train_g, options, options.in_dim)


    if in_nlayers == -1:
        in_nlayers = 0
    if out_nlayers == -1:
        out_nlayers = 0
    sampler = Sampler([None] * (in_nlayers + 1), include_dst_in_src=options.include)

    val_nids = th.tensor(range(val_g.number_of_nodes()))
    print(len(val_nids))
    val_nids = val_nids[val_g.ndata['label_o'].squeeze(-1) != -1]
    print(len(val_nids))
    val_nids1 = val_nids.numpy().tolist()
    shuffle(val_nids1)
    val_nids = val_nids1[:int(len(val_nids1) / 10)]
    test_nids = val_nids1[int(len(val_nids1) / 10):]

    if not os.path.exists(os.path.join(options.model_saving_dir,'val_nids.pkl')):
        with open(os.path.join(options.model_saving_dir,'val_nids.pkl'),'wb') as f:
            pickle.dump(val_nids,f)
        with open(os.path.join(options.model_saving_dir, 'test_nids.pkl'), 'wb') as f:
            pickle.dump(test_nids, f)

    with open(os.path.join('../models/tp6_new1/ln4_bs1024_7/', 'val_nids.pkl'), 'rb') as f:
        val_nids = pickle.load(f)
    with open(os.path.join('../models/tp6_new1/ln4_bs1024_7/', 'test_nids.pkl'), 'rb') as f:
        test_nids = pickle.load(f)
    # create dataloader for training/validate dataset
    traindataloader = MyNodeDataLoader(
        False,
        train_g,
        train_nodes,
        sampler,
        batch_size=options.batch_size,
        shuffle=True,
        drop_last=False,
    )

    valdataloader = MyNodeDataLoader(
        True,
        val_g,
        val_nids,
        sampler,
        batch_size=val_g.num_nodes(),
        shuffle=True,
        drop_last=False,
    )
    testdataloader = MyNodeDataLoader(
        True,
        val_g,
        test_nids,
        sampler,
        batch_size=val_g.num_nodes(),
        shuffle=True,
        drop_last=False,
    )
    print(len(val_nids))

    #exit()
    loaders = [valdataloader]

    beta = options.beta
    # set loss function
    Loss = nn.CrossEntropyLoss()
    # set the optimizer
    optim = th.optim.Adam(
        itertools.chain(mlp.parameters(), model.parameters()),

        options.learning_rate, weight_decay=options.weight_decay
    )

    model.train()
    mlp.train()


    print("Start training")
    pre_loss = 100
    stop_score = 0
    max_F1_score = 0

    # start training
    for epoch in range(options.num_epoch):
        runtime = 0

        total_num,total_loss,correct,fn,fp,tn,tp = 0,0.0,0,0,0,0,0
        pos_count , neg_count =0, 0
        pos_embeddings= th.tensor([]).to(device)
        for ni, (central_nodes,input_nodes,blocks) in enumerate(traindataloader):
            if ni == len(traindataloader)-1:
                continue
            start_time = time()
            # put the block to device
            blocks = [b.to(device) for b in blocks]

            # get in input features
            if options.gnn:
                input_features = blocks[0].srcdata["ntype"]
            else:
                input_features = blocks[0].srcdata["f_input"]
            # the central nodes are the output of the final block
            output_labels = blocks[-1].dstdata[label_name].squeeze(1)
            total_num += len(output_labels)
            # get the embeddings of central nodes
            embedding = model(blocks,input_features)

            # feed the embeddings into the mlp to predict the labels
            label_hat = mlp(embedding)
            if get_options().nlabels != 1:
                pos_prob = nn.functional.softmax(label_hat, 1)[:, 1]
            else:
                pos_prob = th.sigmoid(label_hat)
            # adjust the predicted labels based on a given thredshold beta
            pos_prob[pos_prob >= beta] = 1
            pos_prob[pos_prob < beta] = 0
            predict_labels = pos_prob

            # calculate the loss
            train_loss = Loss(label_hat, output_labels)
            total_loss += train_loss.item() * len(output_labels)
            endtime = time()
            runtime += endtime - start_time

            correct += (
                    predict_labels == output_labels
            ).sum().item()

            # count fake negatives (fn), true negatives (tp), true negatives (tn), true post
            fn += ((predict_labels == 0) & (output_labels != 0)).sum().item()
            tp += ((predict_labels != 0) & (output_labels != 0)).sum().item()
            tn += ((predict_labels == 0) & (output_labels == 0)).sum().item()
            fp += ((predict_labels != 0) & (output_labels == 0)).sum().item()

            start_time = time()
            optim.zero_grad()
            train_loss.backward()
            optim.step()
            endtime = time()
            runtime += endtime-start_time

        Train_loss = total_loss / total_num

        if Train_loss > pre_loss:
            stop_score += 1
            if stop_score >= 2:
                print('Early Stop!')
                #exit()
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
        print("training runtime: ",runtime)
        print("  train:")
        print("\ttp:", tp, " fp:", fp, " fn:", fn, " tn:", tn, " precision:", round(Train_precision,3))
        print("\tloss:{:.8f}, acc:{:.3f}, recall:{:.3f}, F1 score:{:.3f}".format(Train_loss,Train_acc,Train_recall,Train_F1_score))

        # validate
        print("  validate:")
        val_loss, val_acc, val_recall, val_precision, val_F1_score = validate(loaders,label_name, device, model,
                                                                              mlp, Loss, beta,options)
        print("  test:")
        validate([testdataloader], label_name, device, model,
                 mlp, Loss, beta, options)
        # save the result of current epoch
        with open(os.path.join(options.model_saving_dir, 'res.txt'), 'a') as f:
            f.write(str(round(Train_loss, 8)) + " " + str(round(Train_acc, 3)) + " " + str(
                round(Train_recall, 3)) + " " + str(round(Train_precision,3))+" " + str(round(Train_F1_score, 3)) + "\n")
            f.write(str(round(val_loss, 3)) + " " + str(round(val_acc, 3)) + " " + str(
                round(val_recall, 3)) + " "+ str(round(val_precision,3))+" " + str(round(val_F1_score, 3)) + "\n")
            f.write('\n')

        judgement = val_F1_score > max_F1_score
        if judgement:
           max_F1_score = val_F1_score
           print("Saving model.... ", os.path.join(options.model_saving_dir))
           if os.path.exists(options.model_saving_dir) is False:
              os.makedirs(options.model_saving_dir)
           with open(os.path.join(options.model_saving_dir, 'model.pkl'), 'wb') as f:
              parameters = options
              pickle.dump((parameters, model,mlp), f)
           print("Model successfully saved")




if __name__ == "__main__":
    seed = 1234
    # th.set_deterministic(True)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    np.random.seed(seed)
    train(get_options())
