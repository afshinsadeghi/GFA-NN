#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import random
import networkx as nx
import numpy as np
import torch

from torch.utils.data import DataLoader
from model_with5_gpu import KGEModel

from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--use_adadelta_optim', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')
    
    parser.add_argument('--countries', action='store_true', help='Use Countries S1/S2/S3 datasets')
    parser.add_argument('--regions', type=int, nargs='+', default=None, 
                        help='Region Id for Countries S1/S2/S3 datasets, DO NOT MANUALLY SET')
    
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')

    parser.add_argument('--triples_are_mapped', action='store_true')

    parser.add_argument('-node_feat_path', type=str, default=None)

    parser.add_argument('-c0', '--centrality_multiplier0', default=2.0, type=float)
    parser.add_argument('-c1', '--centrality_multiplier1', default=2.0, type=float)
    parser.add_argument('-c2', '--centrality_multiplier2', default=2.0, type=float)
    parser.add_argument('-c3', '--centrality_multiplier3', default=2.0, type=float)
    parser.add_argument('-c4', '--centrality_multiplier4', default=2.0, type=float)

    parser.add_argument('-psi', '--psi', default=14.0, type=float)

    parser.add_argument('--mde_score', action='store_true')
    parser.add_argument('-gamma_1', '--gamma_1', default=2, type=int)
    parser.add_argument('-gamma_2', '--gamma_2', default=2, type=int)
    parser.add_argument('-beta_1', '--beta_1', default=1, type=int)
    parser.add_argument('-beta_2', '--beta_2', default=1, type=int)

    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true', 
                        help='Otherwise use subsampling weighting like in word2vec')
    
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)

    
    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')

    parser.add_argument('--anchor_node_num', type=int, default=1, help='anchor nodes number')
    
    return parser.parse_args(args)

def override_config(args):
    '''
    Override model and data configuration
    '''
    
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    
    args.countries = argparse_dict['countries']
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']
    
def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )
    if args.model == 'MDE':
        entity_embedding = model.entity_embedding0.detach().cpu().numpy()
        np.save(
            os.path.join(args.save_path, 'entity_embedding0'),
            entity_embedding
        )



def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))

def read_mapped_triples(file_path, args):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id_mapped = dict()
        new_eid = 0
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id_mapped[eid] = int(new_eid)
            new_eid = new_eid + 1

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id_mapped = dict()
        new_rid = 0
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id_mapped[rid] = int(new_rid)
            new_rid = new_rid + 1

    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t') #when it is mapped stadardly, it again mapps to the same thing so makes no trouble to use it
            triples.append((int(entity2id_mapped[h]), int(relation2id_mapped[r]), int(entity2id_mapped[t])))
    return triples, entity2id_mapped, relation2id_mapped


def array_norm( array):
    max_dg = torch.max(array).item()
    min_dg = torch.min(array).item()
    if min_dg < 0:
        min_dg = 0
    out_array = (array - min_dg) / (max_dg - min_dg)
    return out_array

#works with extract datafeature_v8.py
def get_node_features(args, entity2id):
    _degree = torch.ones(len(entity2id)) * 10 #this is to include nodes that are not in train. but the entity.dic must be sorted and include train nodes first.
    _degree_ = np.load(args.node_feat_path + "/degree.npy")
    max_dg = np.log((_degree_[:, 1]).astype('int').max()) #np.log(np.max(_degree_[:, 1]))
    _degree[0:_degree_.shape[0]] = torch.log(torch.tensor(_degree_[:, 1].astype('int'), dtype=float)) / max_dg

    _pagerank = torch.ones(len(entity2id)) * 10.0
    _pagerank_ = np.load(args.node_feat_path + "/pagerank.npy", allow_pickle=True, encoding='latin1')
    _pagerank_ = np.array(list(_pagerank_.item().items()))[:, 1]
    _pagerank[0:_pagerank_.shape[0]] = array_norm(torch.tensor(_pagerank_.astype('float')))

    _centrality = torch.ones(len(entity2id)) * 10.0
    _centrality_ = np.load(args.node_feat_path + "/centrality.npy", allow_pickle=True, encoding='latin1')
    _centrality_ = np.array(list(_centrality_.item().items()))[:, 1]
    _centrality[0:_centrality_.shape[0]] = array_norm(torch.tensor(_centrality_.astype('float')))

    #_closeness = torch.ones(len(entity2id)) * 10.0
    #_closeness_ = np.load(args.node_feat_path + "/closeness.npy", allow_pickle=True, encoding='latin1')
    #_closeness_ = np.array(list(_closeness_.item().items()))[:, 1]
    #_closeness[0:_closeness_.shape[0]] = array_norm(torch.tensor(_closeness_))

    _betweenness = torch.ones(len(entity2id)) * 10.0
    _betweenness_ = np.load(args.node_feat_path + "/betweenness.npy", allow_pickle=True, encoding='latin1')
    _betweenness_ = np.array(list(_betweenness_.item().items()))[:, 1]
    _betweenness[0:_betweenness_.shape[0]] = array_norm(torch.tensor(_betweenness_.astype('float')))

    _katz = torch.ones(len(entity2id)) * 10.0
    _katz_ = np.load(args.node_feat_path + "/katz.npy", allow_pickle=True, encoding='latin1')
    _katz_ = np.array(list(_katz_.item().items()))[:, 1]
    _katz[0:_katz_.shape[0]] = array_norm(torch.tensor(_katz_.astype('float')))

    #length = np.load(args.node_feat_path + "/lengthpaths.npy", allow_pickle=True, encoding='latin1').item()
    random_paths_lenghs_ = np.load(args.node_feat_path + "/selected_lengthpaths.npy", allow_pickle=True, encoding='latin1')
    random_paths_lenghs_ = torch.tensor(random_paths_lenghs_)

    max_path_length = torch.max(random_paths_lenghs_) #* 1.0
    random_paths_lenghs = torch.ones(len(entity2id),random_paths_lenghs_.shape[1]) * max_path_length
    random_paths_lenghs[0:random_paths_lenghs_.shape[0],:]= random_paths_lenghs_

    random_paths_lenghs[random_paths_lenghs == -1] = max_path_length + 1.0
    random_paths_lenghs = random_paths_lenghs * 1.0
    #random_paths_lenghs = torch.log(random_paths_lenghs) / torch.log(max_path_length)
    random_paths_lenghs = random_paths_lenghs / (max_path_length + 1.0)

    return [torch.stack((_degree,_pagerank,_centrality,_betweenness,_katz),dim=1).t(),random_paths_lenghs]


def set_gpu(gpus):
    """
    Sets the GPU to be used for the run

    Parameters
    ----------
    gpus:           List of GPUs to be used for the run

    Returns
    -------

    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    gpus_list = gpus.split(",")
    if len(gpus_list) > 1:
        print("running model on multiple gpus", gpus_list)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        #int_gpu_list = list(map(int, gpus))
        #self.device = torch.device('cpu')  # torch.device('cuda:'+gpus[0])
        return torch.device('cuda:' + gpus_list[0])

    elif gpus != '-1' and torch.cuda.is_available():
        return torch.device('cuda')
        #torch.cuda.set_rng_state(torch.cuda.get_rng_state())
        #torch.backends.cudnn.deterministic = True
    else:
        return torch.device('cpu')

def main(args):
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')
    
    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')
    
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # Write logs to checkpoint and console
    set_logger(args)
    
    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)
    args.entity2id = entity2id

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
    
    # Read regions for Countries S* datasets
    if args.countries:
        regions = list()
        with open(os.path.join(args.data_path, 'regions.list')) as fin:
            for line in fin:
                region = line.strip()
                regions.append(entity2id[region])
        args.regions = regions

    nentity = len(entity2id)
    nrelation = len(relation2id)
    
    args.nentity = nentity
    args.nrelation = nrelation
    
    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)

    if args.triples_are_mapped:
        train_triples, entity2id_new, relation2id_new = read_mapped_triples(os.path.join(args.data_path, 'train.txt'),args)
        logging.info('#train: %d' % len(train_triples))

        valid_triples, entity2id_mapped, relation2id_mapped = read_mapped_triples(os.path.join(args.data_path, 'valid.txt'),args)
        logging.info('#valid: %d' % len(valid_triples))

        test_triples, entity2id_mapped, relation2id_mapped = read_mapped_triples(os.path.join(args.data_path, 'test.txt'),args)
        logging.info('#test: %d' % len(test_triples))
    else:
        train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
        logging.info('#train: %d' % len(train_triples))
        valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
        logging.info('#valid: %d' % len(valid_triples))
        test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
        logging.info('#test: %d' % len(test_triples))

    #n_f = Node  ()
    node_features = get_node_features(args, entity2id)

    #n_f=  get_node_features(args,entity2id)
    #All true triples
    all_true_triples = train_triples + valid_triples + test_triples

     #   args= args,
     #   entity2id = entity2id,
    device = set_gpu('0,1,2,3,4')
    kge_model = KGEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding
    )

    if args.cuda:
        #kge_model = kge_model.cuda()
        kge_model = kge_model.to(device)
        #print("kge_model is cuda", kge_model.is_cuda)

    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))




    kge_model.set_node_features(args, node_features)

    if args.do_train:
        # Set training dataloader iterator
        train_dataloader_head = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'head-batch'),
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )
        
        train_dataloader_tail = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch'),
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )
        
        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
        
        # Set training configuration
        current_learning_rate = args.learning_rate
        if args.use_adadelta_optim :
            optimizer = torch.optim.Adadelta(
                filter(lambda p: p.requires_grad, kge_model.parameters()),
                lr=current_learning_rate, weight_decay=1e-6
            )
        else:

            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, kge_model.parameters()),
                lr=current_learning_rate
            )
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'), map_location='cpu')
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        device = set_gpu('0,1,2,3,4')
        kge_model = kge_model.to(device)

        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_step = 0
    
    step = init_step
    
    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_sample_size = %d' % args.negative_sample_size)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)
    
    # Set valid dataloader as it would be evaluated during training

    if args.do_train:
        if current_learning_rate == 0:
            current_learning_rate = args.learning_rate

        logging.info('learning_rate = %f' % current_learning_rate)

        training_logs = []
        
        #Training Loop
        for step in range(init_step, args.max_steps):
            
            log = kge_model.train_step(kge_model, optimizer, train_iterator, args)
            
            training_logs.append(log)
            
            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))

                if args.use_adadelta_optim:
                    optimizer = torch.optim.Adadelta(
                        filter(lambda p: p.requires_grad, kge_model.parameters()),
                        lr=current_learning_rate, weight_decay=1e-6
                    )
                else:
                    optimizer = torch.optim.Adam(
                        filter(lambda p: p.requires_grad, kge_model.parameters()),
                        lr=current_learning_rate
                    )
                warm_up_steps = warm_up_steps * 3
            
            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step, 
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(kge_model, optimizer, save_variable_list, args)
                
            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []
                
            #if args.do_valid and step > 140000 and step % args.valid_steps == 0:
            #    logging.info('Evaluating on Valid Dataset...')
            #    metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
            #    log_metrics('Valid', step, metrics)

            if args.do_test and step > 20000 and step % args.valid_steps == 0:
                logging.info('Evaluating on Test Dataset...')
                metrics = kge_model.test_step(kge_model, test_triples, all_true_triples, args)
                log_metrics('Test', step, metrics)

        save_variable_list = {
            'step': step, 
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(kge_model, optimizer, save_variable_list, args)
        
    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
        log_metrics('Valid', step, metrics)
    
    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        metrics = kge_model.test_step(kge_model, test_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)
    
    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        metrics = kge_model.test_step(kge_model, train_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)
        
if __name__ == '__main__':
    main(parse_args())
