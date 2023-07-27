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
from MyDataLoader2 import *
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

def get_reverse_graph(g):
    edges = g.edges()
    reverse_edges = (edges[1], edges[0])

    rg = dgl.graph(reverse_edges, num_nodes=g.num_nodes())
    for key, value in g.ndata.items():
        # print(key,value)
        rg.ndata[key] = value
    for key, value in g.edata.items():
        # print(key,value)
        rg.edata[key] = value
    return rg

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

def unlabel_low(g,unlabel_threshold):
    mask_low = g.ndata['position'] <= unlabel_threshold
    g.ndata['adder_o'][mask_low] = 0

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
    if options.mask:
        no_internal_mask = g.ndata['internal'].squeeze() == 0
        mask = th.logical_and(no_muldiv_mask,no_internal_mask)
    else:
        mask = no_muldiv_mask
    print('#nodes after masking',len(labels[mask]))
    nodes = th.tensor(range(g.num_nodes()))
    nodes = nodes[mask]

    labels = labels[mask]
    print(len(nodes))

    mask_pos = (labels ==1).squeeze(1)

    mask_neg = (labels == 0).squeeze(1)
    pos_nodes = nodes[mask_pos].numpy().tolist()
    neg_nodes = nodes[mask_neg].numpy().tolist()
    shuffle(pos_nodes)
    shuffle(neg_nodes)
    pos_size = len(pos_nodes)
    neg_size = len(neg_nodes)
    print(len(g.ndata['position'][g.ndata['adder_o'].squeeze(-1) == 1].numpy().tolist()))
    print(len(pos_nodes))
    ratio = float(neg_size) / float(pos_size)
    print("ratio=", ratio)


    pos_count = th.zeros(size=(1, in_dim+1)).squeeze(0).numpy().tolist()
    neg_count = th.zeros(size=(1, in_dim+1)).squeeze(0).numpy().tolist()
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
    print('----------------Preprocessing----------------')
    label2id = {}
    if os.path.exists(data_path) is False:
        os.makedirs(data_path)
    train_data_file = os.path.join(data_path, 'BOOM.pkl')
    val_data_file = os.path.join(data_path, 'RocketCore.pkl')
    print(val_data_file)
    # generate and save the test dataset if missing
    if os.path.exists(val_data_file) is False:
        print('Validation dataset does not exist. Generating validation dataset... ')
        datapaths = [os.path.join(options.val_netlist_path, 'implementation')]
        report_folders = [os.path.join(options.val_netlist_path, 'report')]
        th.multiprocessing.set_sharing_strategy('file_system')
        dataset = Dataset(options.val_top,datapaths,report_folders,label2id)
        g = dataset.batch_graph
        with open(val_data_file,'wb') as f:
            pickle.dump(g,f)
    print('Validation dataset is ready!')
    # generate and save the train dataset if missing
    if os.path.exists(train_data_file) is False:
        print('Training dataset does not exist. Generating training dataset... ')
        datapaths = [os.path.join(options.train_netlist_path, 'implementation')]
        report_folders = [os.path.join(options.train_netlist_path, 'report')]
        th.multiprocessing.set_sharing_strategy('file_system')
        dataset = Dataset(options.train_top, datapaths, report_folders, label2id)
        g = dataset.batch_graph
        with open(train_data_file,'wb') as f:
            pickle.dump(g,f)
    print('Training dataset is ready!')
    if len(label2id) != 0:
        with open(os.path.join(data_path,'label2id.pkl'),'wb') as f:
            pickle.dump(label2id,f)
    if options.pre_train:
        with open(os.path.join(options.pre_model_dir,'model.pkl'),'rb') as f:
                _, classifier,_ = pickle.load(f)
        print("Loading model from:", options.pre_model_dir)
        print(classifier)
    else:
        print('Intializing models...')
    # initialize the  model
        if options.sage:
            network = GraphSage
        elif options.abgnn:
            network = ABGNN
        elif options.function:
            network = FuncGNN
        else:
            print('please choose a valid model type!')
            exit()

        if options.in_nlayers!=-1:
            model1 = network(
                ntypes = options.in_dim,
                hidden_dim=options.hidden_dim,
                out_dim=options.out_dim,
                n_layers = options.in_nlayers,
                in_dim = options.in_dim,
                dropout=options.gcn_dropout,
            )
            out_dim1 = model1.out_dim
        else:
            model1 = None
            out_dim1 = 0
        if options.out_nlayers!=-1:
            model2 = network(
                ntypes=options.in_dim,
                hidden_dim=options.hidden_dim,
                out_dim=options.out_dim,
                n_layers=options.out_nlayers,
                in_dim=options.in_dim,
                dropout=options.gcn_dropout,
            )
            out_dim2 = model2.out_dim
        else:
            model2 = None
            out_dim2 =0
        # initialize a multlayer perceptron

        mlp = MLP(
            in_dim = out_dim1+out_dim2,
            out_dim = options.nlabels,
            nlayers = options.n_fcn,
            dropout = options.mlp_dropout
        ).to(device)

        classifier = BiClassifier(model1,model2,mlp)
        print(classifier)
        print("creating model in:",options.model_saving_dir)
    # save the model and create a file a save the results
    if os.path.exists(options.model_saving_dir) is False:
        os.makedirs(options.model_saving_dir)
        with open(os.path.join(options.model_saving_dir, 'model.pkl'), 'wb') as f:
            parameters = options
            pickle.dump((parameters, classifier), f)
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
        param, classifier = pickle.load(f)
        param.model_saving_dir = options.model_saving_dir
        classifier = classifier.to(device)

        # make some changes to the options
        if options.change_lr:
            param.learning_rate = options.learning_rate
        if options.change_alpha:
            param.alpha = options.alpha
    return param,classifier



def validate(loaders,label_name,device,model,Loss,beta,options):
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
            for ni, (in_blocks,out_blocks) in enumerate(loader):
                start = time()
                in_blocks = [b.to(device) for b in in_blocks]
                out_blocks = [b.to(device) for b in out_blocks]
                #print(out_blocks)
                # get in input features
                if not options.function:
                    in_input_features = in_blocks[0].srcdata["ntype"]
                    out_input_features = out_blocks[0].srcdata["ntype"]
                else:
                    in_input_features = in_blocks[0].srcdata["h"]
                    out_input_features = out_blocks[0].srcdata["h"]
                # the central nodes are the output of the final block
                output_labels = in_blocks[-1].dstdata[label_name].squeeze(1)
                total_num += len(output_labels)
                # predict the labels of central nodes
                label_hat = model(in_blocks, in_input_features, out_blocks, out_input_features)
                # if options.abgnn:
                #     label_hat = model(in_blocks, in_input_features, out_blocks, out_input_features)
                # else:
                #     label_hat = model(in_blocks, in_input_features)
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
    train_data_file = os.path.join(data_path,'BOOM.pkl')
    val_data_file = os.path.join(data_path,'RocketCore.pkl')

    # preprocess: generate dataset / initialize the model
    if options.preprocess :
        preprocess(data_path,device,options)
        return
    print(options)
    # load the model
    options, model = load_model(device, options)
    if model is None:
        print("No model, please prepocess first , or choose a pretrain model")
        return
    print(model)

    in_nlayers = options.in_nlayers if isinstance(options.in_nlayers,int) else options.in_nlayers[0]
    out_nlayers = options.out_nlayers if isinstance(options.out_nlayers,int) else options.out_nlayers[0]

    label_name = 'adder_o'
    print("----------------Loading data----------------")
    with open(train_data_file,'rb') as f:
        train_g,_ = pickle.load(f)
        if options.function:
            train_g.ndata['h'] = th.ones((train_g.number_of_nodes(),options.hidden_dim),dtype=th.float)
        #train_g.ndata['label_o'] = train_g.ndata['adder_o']
        train_g.ndata['ntype2'] = th.argmax(train_g.ndata['ntype'], dim=1).squeeze(-1)
        train_g.ndata['temp'] = th.ones(size=(train_g.number_of_nodes(), options.hidden_dim),
                                      dtype=th.float)
        train_g.ndata['adder_o'][train_g.ndata['mul_o']==1] = -1
        #train_g.ndata['adder_o'][train_g.ndata['sub_o'] == 1] = -1
        train_graphs = dgl.unbatch(train_g)
        if options.train_percent == 1:
            train_graphs = [train_graphs[3]]
        else:
            train_graphs = train_graphs[:int(options.train_percent)]
        train_g = dgl.batch(train_graphs)
    with open(val_data_file,'rb') as f:
        val_g,_ = pickle.load(f)
        #val_g.ndata['label_o'] = val_g.ndata['adder_o']
        val_g.ndata['ntype2'] = th.argmax(val_g.ndata['ntype'], dim=1).squeeze(-1)
        print(len(val_g.ndata['adder_o'][val_g.ndata['adder_o'] == 1]))
        print(len(val_g.ndata['adder_o'][val_g.ndata['mul_o'] == 1]))
        val_g.ndata['adder_o'][val_g.ndata['mul_o'] == 1] = -1
        #val_g.ndata['adder_o'][val_g.ndata['sub_o'] == 1] = -1
        print(len(val_g.ndata['adder_o'][val_g.ndata['adder_o'] == 1]))
        #exit()
        if options.function:
            val_g.ndata['h'] = th.ones((val_g.number_of_nodes(),options.hidden_dim),dtype=th.float)
            val_g.ndata['temp'] = th.ones(size=(val_g.number_of_nodes(), options.hidden_dim),
                                          dtype=th.float)

    #print(train_g.ndata['position'][train_g.ndata['adder_o'].squeeze(-1) == 1].numpy().tolist())
    train_g.ndata['position'][train_g.ndata['adder_o'].squeeze(-1) == -1] = 100
    val_g.ndata['position'][val_g.ndata['adder_o'].squeeze(-1) == -1] = 100
    unlabel_low(train_g, options.unlabel)
    unlabel_low(val_g, options.unlabel)
    #print(train_g.ndata['position'][train_g.ndata['adder_o'].squeeze(-1) == 1].numpy().tolist())
    #print(len())
    train_nodes, pos_count, neg_count = oversample(train_g, options, options.in_dim)

    if in_nlayers == -1:
        in_nlayers = 0
    if out_nlayers == -1:
        out_nlayers = 0
    in_sampler = Sampler([None] * (in_nlayers + 1), include_dst_in_src=options.include)
    out_sampler = Sampler([None] * (out_nlayers + 1), include_dst_in_src=options.include)

    if options.mask:
        val_split_file = "val_nids_masked_nodiv.pkl"
        test_split_file = "test_nids_masked_nodiv.pkl"
    else:
        val_split_file = "val_nids.pkl"
        test_split_file = "test_nids.pkl"
    if os.path.exists(os.path.join(options.datapath, val_split_file)):
        with open(os.path.join(options.datapath, val_split_file), 'rb') as f:
            val_nids = pickle.load(f)
        with open(os.path.join(options.datapath, test_split_file), 'rb') as f:
            test_nids = pickle.load(f)
    else:
        nids = th.tensor(range(val_g.number_of_nodes()))
        mask1 = val_g.ndata['adder_o'].squeeze(-1) != -1
        if options.mask:
            mask2 = val_g.ndata['internal'].squeeze() == 0
            mask = th.logical_and(mask1,mask2)
        else:
            mask = mask1
        nids = nids[mask]
        nids = nids.numpy().tolist()
        shuffle(nids)
        val_nids = nids[:int(len(nids) / 10)]
        test_nids = nids[int(len(nids) / 10):]

        with open(os.path.join(options.datapath, val_split_file), 'wb') as f:
            pickle.dump(val_nids, f)
        with open(os.path.join(options.datapath, test_split_file), 'wb') as f:
            pickle.dump(test_nids, f)


    # create dataloader for training/validate dataset
    if options.sage:
        graph_function = DAG2UDG
        out_sampler.include_dst_in_src = True
    else:
        graph_function = get_reverse_graph

    traindataloader = MyNodeDataLoader(
        False,
        train_g,
        graph_function(train_g),
        train_nodes,
        in_sampler,
        out_sampler,
        batch_size=options.batch_size,
        shuffle=True,
        drop_last=False,
    )
    valdataloader = MyNodeDataLoader(
        True,
        val_g,
        graph_function(val_g),
        val_nids,
        in_sampler,
        out_sampler,
        batch_size=val_g.num_nodes(),
        shuffle=True,
        drop_last=False,
    )
    testdataloader = MyNodeDataLoader(
        True,
        val_g,
        graph_function(val_g),
        test_nids,
        in_sampler,
        out_sampler,
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
        model.parameters(), options.learning_rate, weight_decay=options.weight_decay
    )
    model.train()
    if options.abgnn:
        if model.GCN1 is not None: model.GCN1.train()
        if model.GCN2 is not None: model.GCN2.train()


    print("----------------Start training----------------")
    pre_loss = 100
    stop_score = 0
    max_F1_score = 0

    # start training
    for epoch in range(options.num_epoch):
        runtime = 0

        total_num,total_loss,correct,fn,fp,tn,tp = 0,0.0,0,0,0,0,0
        pos_count , neg_count =0, 0
        pos_embeddings= th.tensor([]).to(device)
        for ni, (in_blocks,out_blocks) in enumerate(traindataloader):
            if ni == len(traindataloader)-1:
                continue
            start_time = time()
            #print(out_blocks)
            # put the block to device
            in_blocks = [b.to(device) for b in in_blocks]
            out_blocks = [b.to(device) for b in out_blocks]
            # get in input features
            if not options.function:
                in_input_features = in_blocks[0].srcdata["ntype"]
                out_input_features = out_blocks[0].srcdata["ntype"]
            else:
                in_input_features = in_blocks[0].srcdata["h"]
                out_input_features = out_blocks[0].srcdata["h"]
            # if epoch>0:
            #     print(in_input_features)
            # the central nodes are the output of the final block
            output_labels = in_blocks[-1].dstdata[label_name].squeeze(1)
            total_num += len(output_labels)
            # predict the labels of central nodes

            label_hat = model(in_blocks, in_input_features, out_blocks, out_input_features)
            # if options.abgnn:
            #     label_hat = model(in_blocks,in_input_features,out_blocks,out_input_features)
            # else:
            #     label_hat = model(in_blocks, in_input_features)
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
                                                                              Loss, beta,options)
        print("  test:")
        validate([testdataloader], label_name, device, model,
                 Loss, beta, options)
        # save the result of current epoch
        print("----------------Saving the results---------------")
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
              pickle.dump((parameters, model), f)
           print("Model successfully saved")




if __name__ == "__main__":
    seed = 1234
    # th.set_deterministic(True)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    np.random.seed(seed)
    train(get_options())
