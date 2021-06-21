#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

from dataloader import TestDataset
import networkx as nx
import random

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma,
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.mde_vector_number = 8
        self.node_features = torch.tensor([])
        self.zero_f_tensor = torch.zeros(6)


        self.w = nn.Parameter(
            (torch.FloatTensor(np.random.uniform(-1 / 5, 1 / 5, [10]))))

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim * 2 if double_relation_embedding else hidden_dim

        if model_name == 'Rotate':
            self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
            nn.init.uniform_(
                tensor=self.entity_embedding,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )

            self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
            nn.init.uniform_(
                tensor=self.relation_embedding,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if model_name == 'MDE':
            self.entity_embedding0 = nn.Parameter(torch.zeros(nentity, self.entity_dim, device=device))
            self.entity_embedding1 = nn.Parameter(torch.zeros(nentity, self.entity_dim, device=device))
            #self.entity_embedding2 = nn.Parameter(torch.zeros(nentity, self.entity_dim, device=device))
            self.entity_embedding3 = nn.Parameter(torch.zeros(nentity, self.entity_dim*3, device=device))
            self.entity_embedding4 = nn.Parameter(torch.zeros(nentity, self.entity_dim, device=device))
            self.entity_embedding5 = nn.Parameter(torch.zeros(nentity, self.entity_dim, device=device))
            #self.entity_embedding6 = nn.Parameter(torch.zeros(nentity, self.entity_dim, device=device))
            self.entity_embedding7 = nn.Parameter(torch.zeros(nentity, self.entity_dim*3, device=device))
            self.entity_embedding8 = nn.Parameter(torch.zeros(nentity, self.entity_dim, device=device))
            self.entity_embedding9 = nn.Parameter(torch.zeros(nentity, self.entity_dim, device=device))
            self.entity_embedding10 = nn.Parameter(torch.zeros(nentity, self.entity_dim, device=device))
            self.entity_embedding11 = nn.Parameter(torch.zeros(nentity, self.entity_dim, device=device))
            self.entity_embedding12 = nn.Parameter(torch.zeros(nentity, self.entity_dim*3, device=device))
            nn.init.uniform_(
                tensor=self.entity_embedding0,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )

            nn.init.uniform_(
                tensor=self.entity_embedding1,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            # nn.init.uniform_(
            #     tensor=self.entity_embedding2,
            #     a=-self.embedding_range.item(),
            #     b=self.embedding_range.item()
            # )
            nn.init.uniform_(
                tensor=self.entity_embedding3,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.entity_embedding4,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.entity_embedding5,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            # nn.init.uniform_(
            #     tensor=self.entity_embedding6,
            #     a=-self.embedding_range.item(),
            #     b=self.embedding_range.item()
            # )
            nn.init.uniform_(
                tensor=self.entity_embedding7,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
             )
            # nn.init.uniform_(
            #     tensor=self.entity_embedding8,
            #     a=-self.embedding_range.item(),
            #     b=self.embedding_range.item()
            # )
            nn.init.uniform_(
                tensor=self.entity_embedding9,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.entity_embedding10,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            # nn.init.uniform_(
            #     tensor=self.entity_embedding11,
            #     a=-self.embedding_range.item(),
            #     b=self.embedding_range.item()
            # )
            # nn.init.uniform_(
            #     tensor=self.entity_embedding12,
            #     a=-self.embedding_range.item(),
            #     b=self.embedding_range.item()
            # )
            self.relation_embedding0 = nn.Parameter(torch.zeros(nrelation, self.relation_dim, device=device))
            self.relation_embedding1 = nn.Parameter(torch.zeros(nrelation, self.relation_dim, device=device))
            self.relation_embedding2 = nn.Parameter(torch.zeros(nrelation, self.relation_dim*3, device=device))
            self.relation_embedding3 = nn.Parameter(torch.zeros(nrelation, self.relation_dim, device=device))
            self.relation_embedding4 = nn.Parameter(torch.zeros(nrelation, self.relation_dim, device=device))
            self.relation_embedding5 = nn.Parameter(torch.zeros(nrelation, self.relation_dim*3, device=device))
            self.relation_embedding6 = nn.Parameter(torch.zeros(nrelation, self.relation_dim*3, device=device)) #for paths lenghts
            #self.relation_embedding7 = nn.Parameter(torch.zeros(nrelation, self.relation_dim*3, device=device))

            nn.init.uniform_(
                tensor=self.relation_embedding0,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )

            nn.init.uniform_(
                tensor=self.relation_embedding1,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.relation_embedding2,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.relation_embedding3,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.relation_embedding4,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.relation_embedding5,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.relation_embedding6,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            # nn.init.uniform_(
            #     tensor=self.relation_embedding7,
            #     a=-self.embedding_range.item(),
            #     b=self.embedding_range.item()
            # )

        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))

        # Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE', 'MDE']:
            raise ValueError('model %s not supported' % model_name)

        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')

    def set_node_features(self, args, node_features):
        self.args = args
        self.node_features = node_features[0]
        #self.graph = MG
        max_used_distances = 3000
        dim_max = min(self.entity_dim*3,max_used_distances )
        self.lenghts = node_features[1][:,0:dim_max]
        self.anchor_node_num = args.anchor_node_num
        self.max_graph_diameter = 30
        self.unreachable_node_distance = self.max_graph_diameter
        self.shortest_path_div = (self.max_graph_diameter * self.anchor_node_num )
        if args.cuda:
            self.node_features = node_features[0].cuda()
            self.lenghts = node_features[1][:,0:dim_max].cuda()



    # def short_path_length(self,node1, node2):
    #     try:
    #         return self.lenghts[node1][node2]/ self.max_distance#nx.shortest_path_length(self.graph, node1, node2, weight=1) / self.max_distance
    #     except:
    #         return 1.0

    # def batch_short_path_length(self,node1_list, node2):
    #     result_list = []
    #     for node in node1_list:
    #         result = self.short_path_length(node.item(),node2)
    #         result_list.append(result)
    #     return torch.tensor(result_list)


    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        '''
        #print(sample)
        head_part = []
        tail_part = []
        sampled_nodes = []

        if mode == 'single':
            head_part = sample[:, 0]
            tail_part = sample[:, 2]
            indice = random.sample(range(head_part.size(0)), 1 )#self.anchor_node_num
            indice = torch.tensor(indice)
            sampled_nodes = head_part[indice]
            #print(indice)
            #print(tail_part.size())
            #print(head_part.size()) #
            #print(sampled_nodes)
            #print(head_part)
            #print(tail_part)

        elif mode == 'head-batch':
            tail_part, head_part  = sample
            tail_part = tail_part[:, 2]
            head_part = head_part.view(-1)#head_part[0] #these are corrupted batch*negative_sample size
            #print(tail_part.size())
            #print(head_part.size())
            indice = random.sample(range(tail_part.size(0)), 1) ##self.anchor_node_num
            #print(indice)
            indice = torch.tensor(indice)
            sampled_nodes = tail_part[indice]

            #print(sampled_nodes)
            #print(head_part)
            #print(tail_part)

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            head_part = head_part[:, 0]
            #tail_part = tail_part[0] #it must be torch.Size([210, 624]) 210* 624, if not some thing is wrong. no it is 620!
            tail_part = tail_part.view(-1)
            indice = random.sample(range(head_part.size(0)), 1 )#self.anchor_node_num
            #print(indice)
            indice = torch.tensor(indice)
            sampled_nodes = head_part[indice]

            #print(tail_part.size())
            #print(head_part.size())

        #distances_h = torch.tensor(np.vectorize(self.short_path_length)(head_part.cpu(), sampled_nodes[0].cpu()))
        #distances_t = torch.tensor(np.vectorize(self.short_path_length)(tail_part.cpu(), sampled_nodes[0].cpu()))
        #print("self.anchor_node_num line 300", self.anchor_node_num )


        #this is the verison with more than 1 anchor node
        #if self.anchor_node_num == 1:
        #    distances_h = self.batch_short_path_length_new(head_part,sampled_nodes)
        #    distances_t = self.batch_short_path_length_new(tail_part, sampled_nodes)
        #else:
        #distances_h = self.batch_short_path_length(head_part, sampled_nodes[0].item())
        #distances_t = self.batch_short_path_length(tail_part, sampled_nodes[0].item())
        #print(head_part)
        distances_h= self.lenghts[head_part]
        #print(distances_h)
        #print(tail_part)
        #print(self.lenghts.size())
        distances_t = self.lenghts[tail_part]
        #print(distances_t)
        if self.args.cuda:
            distances_h = distances_h.cuda()
            distances_t = distances_t.cuda()
        #print(mode)
        #print(sampled_nodes)
            #print(head_part)
            #print(tail_part)

        #print(distances_h)
        #print(distances_t)
        #print(self.args.entity2id)

        #print(sampled_nodes)
        #print(nx.shortest_path_length(self.graph, 10167, 27433))

        #distances_h = nx.shortest_path_length(self.graph, head_part[:,0][0], sampled_nodes[0].item())
        #distances_t = nx.shortest_path_length(self.graph, tail_part[0], sampled_nodes)


        if mode == 'single' and self.model_name != 'MDE':
            batch_size, negative_sample_size = sample.size(0), 1
            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)
        elif mode == 'single' and self.model_name == 'MDE':
            batch_size, negative_sample_size = sample.size(0), 1
            h0 = torch.index_select(
                self.entity_embedding0,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            r0 = torch.index_select(
                self.relation_embedding0,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            t0 = torch.index_select(
                self.entity_embedding0,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

            h1 = torch.index_select(
                self.entity_embedding1,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            r1 = torch.index_select(
                self.relation_embedding1,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            t1 = torch.index_select(
                self.entity_embedding1,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

            # h2 = torch.index_select(
            #     self.entity_embedding2,
            #     dim=0,
            #     index=sample[:, 0]
            # ).unsqueeze(1)
            #
            r2 = torch.index_select(
                self.relation_embedding2,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)
            #
            # t2 = torch.index_select(
            #     self.entity_embedding2,
            #     dim=0,
            #     index=sample[:, 2]
            # ).unsqueeze(1)

            h3 = torch.index_select(
                self.entity_embedding3,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            r3 = torch.index_select(
                self.relation_embedding3,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            t3 = torch.index_select(
                self.entity_embedding3,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

            h4 = torch.index_select(
                self.entity_embedding4,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            r4 = torch.index_select(
                self.relation_embedding4,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            t4 = torch.index_select(
                self.entity_embedding4,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

            h5 = torch.index_select(
                self.entity_embedding5,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            r5 = torch.index_select(
                self.relation_embedding5,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            t5 = torch.index_select(
                self.entity_embedding5,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

            # h6 = torch.index_select(
            #     self.entity_embedding6,
            #     dim=0,
            #     index=sample[:, 0]
            # ).unsqueeze(1)
            #
            r6 = torch.index_select(
                self.relation_embedding6,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            # t6 = torch.index_select(
            #     self.entity_embedding6,
            #     dim=0,
            #     index=sample[:, 2]
            # ).unsqueeze(1)

            h7 = torch.index_select(
                self.entity_embedding7,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            # r7 = torch.index_select(
            #     self.relation_embedding7,
            #     dim=0,
            #     index=sample[:, 1]
            # ).unsqueeze(1)

            t7 = torch.index_select(
                self.entity_embedding7,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

            h8 = torch.index_select(
                self.entity_embedding8,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            t8 = torch.index_select(
                self.entity_embedding8,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

            h9 = torch.index_select(
                self.entity_embedding9,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            t9 = torch.index_select(
                self.entity_embedding9,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

            h10 = torch.index_select(
                self.entity_embedding10,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            t10 = torch.index_select(
                self.entity_embedding10,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

            h11 = torch.index_select(
                self.entity_embedding11,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            t11 = torch.index_select(
                self.entity_embedding11,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

            h12 = torch.index_select(
                self.entity_embedding12,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            t12 = torch.index_select(
                self.entity_embedding12,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)
            # _degree,_pagerank,_centrality,_closeness,_betweenness,_katz
            # h_d = self.node_features[0][sample[:, 0]].view(batch_size, -1)
            # t_d = self.node_features[0][sample[:, 2]].view(batch_size, -1)
            # h_d = self.zero_f_tensor.clone()#torch.zeros(6)
            # t_d = self.zero_f_tensor.clone()
            h_d = []  # self.zero_f_tensor.clone()#torch.zeros(6)
            t_d = []  # self.zero_f_tensor.clone()#torch.zeros(6)
            for i in range(0, 5):
                h_d_ = self.node_features[i][sample[:, 0]].view(batch_size, -1)
                t_d_ = self.node_features[i][sample[:, 2]].view(batch_size, -1)
                h_d.append(h_d_)
                t_d.append(t_d_)
            #distances_h = distances_h.view(batch_size, -1)
            #distances_t = distances_t.view(batch_size, -1)
            #todo:replace the h_d_[3] (the 4th one which is _betweenness (_closeness is already removed))


            head = [h0, h1,  h3, h4, h5, h7, h8, h9, h10, h11, h12, h_d, distances_h]#h2, h6,
            relation = [r0, r1,r2, r3, r4, r5, r6]# r7
            tail = [t0, t1,  t3, t4, t5,  t7, t8, t9, t10, t11, t12, t_d,distances_t]#t2,t6,

        elif mode == 'head-batch' and self.model_name != 'MDE':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode == 'tail-batch' and self.model_name != 'MDE':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        elif mode == 'head-batch' and self.model_name == 'MDE':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            h0 = torch.index_select(
                self.entity_embedding0,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            r0 = torch.index_select(
                self.relation_embedding0,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            t0 = torch.index_select(
                self.entity_embedding0,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

            h1 = torch.index_select(
                self.entity_embedding1,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            r1 = torch.index_select(
                self.relation_embedding1,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            t1 = torch.index_select(
                self.entity_embedding1,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

            # h2 = torch.index_select(
            #     self.entity_embedding2,
            #     dim=0,
            #     index=head_part.view(-1)
            # ).view(batch_size, negative_sample_size, -1)
            #
            r2 = torch.index_select(
                self.relation_embedding2,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)
            #
            # t2 = torch.index_select(
            #     self.entity_embedding2,
            #     dim=0,
            #     index=tail_part[:, 2]
            # ).unsqueeze(1)

            h3 = torch.index_select(
                self.entity_embedding3,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            r3 = torch.index_select(
                self.relation_embedding3,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            t3 = torch.index_select(
                self.entity_embedding3,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

            h4 = torch.index_select(
                self.entity_embedding4,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            r4 = torch.index_select(
                self.relation_embedding4,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            t4 = torch.index_select(
                self.entity_embedding4,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

            h5 = torch.index_select(
                self.entity_embedding5,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            r5 = torch.index_select(
                self.relation_embedding5,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            t5 = torch.index_select(
                self.entity_embedding5,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

            # h6 = torch.index_select(
            #     self.entity_embedding6,
            #     dim=0,
            #     index=head_part.view(-1)
            # ).view(batch_size, negative_sample_size, -1)
            #
            r6 = torch.index_select(
                self.relation_embedding6,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)
            #
            # t6 = torch.index_select(
            #     self.entity_embedding6,
            #     dim=0,
            #     index=tail_part[:, 2]
            # ).unsqueeze(1)

            h7 = torch.index_select(
                self.entity_embedding7,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            # r7 = torch.index_select(
            #     self.relation_embedding7,
            #     dim=0,
            #     index=tail_part[:, 1]
            # ).unsqueeze(1)

            t7 = torch.index_select(
                self.entity_embedding7,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

            h8 = torch.index_select(
                self.entity_embedding8,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            t8 = torch.index_select(
                self.entity_embedding8,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

            h9 = torch.index_select(
                self.entity_embedding9,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            t9 = torch.index_select(
                self.entity_embedding9,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

            h10 = torch.index_select(
                self.entity_embedding10,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            t10 = torch.index_select(
                self.entity_embedding10,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

            h11 = torch.index_select(
                self.entity_embedding11,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            t11 = torch.index_select(
                self.entity_embedding11,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

            h12 = torch.index_select(
                self.entity_embedding12,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            t12 = torch.index_select(
                self.entity_embedding12,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)
            # h_d = self.node_features[0][head_part.view(-1)]
            # print(h_d.shape)
            # print(t_d.shape)
            # h_d = torch.index_select(self.node_features[0], dim=0, index=head_part.view(-1)).view(batch_size, negative_sample_size)
            # t_d = self.node_features[0][tail_part[:, 2]].view(batch_size, -1)
            # h_d = self.zero_f_tensor.clone()#torch.zeros(6)
            # t_d = self.zero_f_tensor.clone()#torch.zeros(6)
            h_d = []  # self.zero_f_tensor.clone()#torch.zeros(6)
            t_d = []  # self.zero_f_tensor.clone()#torch.zeros(6)
            for i in range(0, 5):
                h_d_ = torch.index_select(self.node_features[i], dim=0, index=head_part.view(-1)).view(batch_size,
                                                                                                       negative_sample_size)
                t_d_ = self.node_features[i][tail_part[:, 2]].view(batch_size, -1)
                h_d.append(h_d_)
                t_d.append(t_d_)
            # for i in range(0, 6):
            #    h_d[i] = torch.index_select(self.node_features[i], dim=0, index=head_part.view(-1)).view(batch_size, negative_sample_size)
            #    t_d[i] = self.node_features[i][tail_part[:, 2]].view(batch_size, -1)

            #distances_h = distances_h.view(batch_size, negative_sample_size)
            #distances_t = distances_t.view(batch_size, -1)

            head = [h0, h1, h3, h4, h5, h7, h8, h9, h10, h11, h12, h_d, distances_h]# h2,  h6,
            relation = [r0, r1,r2, r3, r4, r5, r6]# r7
            tail = [t0, t1, t3, t4, t5, t7, t8, t9, t10, t11, t12, t_d, distances_t]#t2, t6,

        elif mode == 'tail-batch' and self.model_name == 'MDE':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            h0 = torch.index_select(
                self.entity_embedding0,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            r0 = torch.index_select(
                self.relation_embedding0,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            t0 = torch.index_select(
                self.entity_embedding0,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            h1 = torch.index_select(
                self.entity_embedding1,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            r1 = torch.index_select(
                self.relation_embedding1,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            t1 = torch.index_select(
                self.entity_embedding1,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            # h2 = torch.index_select(
            #     self.entity_embedding2,
            #     dim=0,
            #     index=head_part[:, 0]
            # ).unsqueeze(1)
            #
            r2 = torch.index_select(
                self.relation_embedding2,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
            #
            # t2 = torch.index_select(
            #     self.entity_embedding2,
            #     dim=0,
            #     index=tail_part.view(-1)
            # ).view(batch_size, negative_sample_size, -1)
            h3 = torch.index_select(
                self.entity_embedding3,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            r3 = torch.index_select(
                self.relation_embedding3,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            t3 = torch.index_select(
                self.entity_embedding3,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            h4 = torch.index_select(
                self.entity_embedding4,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            r4 = torch.index_select(
                self.relation_embedding4,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            t4 = torch.index_select(
                self.entity_embedding4,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            h5 = torch.index_select(
                self.entity_embedding5,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            r5 = torch.index_select(
                self.relation_embedding5,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            t5 = torch.index_select(
                self.entity_embedding5,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            # h6 = torch.index_select(
            #     self.entity_embedding6,
            #     dim=0,
            #     index=head_part[:, 0]
            # ).unsqueeze(1)
            #
            r6 = torch.index_select(
                self.relation_embedding6,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            # t6 = torch.index_select(
            #     self.entity_embedding6,
            #     dim=0,
            #     index=tail_part.view(-1)
            # ).view(batch_size, negative_sample_size, -1)
            h7 = torch.index_select(
                self.entity_embedding7,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            # r7 = torch.index_select(
            #     self.relation_embedding7,
            #     dim=0,
            #     index=head_part[:, 1]
            # ).unsqueeze(1)

            t7 = torch.index_select(
                self.entity_embedding7,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            h8 = torch.index_select(
                self.entity_embedding8,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            t8 = torch.index_select(
                self.entity_embedding8,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            h9 = torch.index_select(
                self.entity_embedding9,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            t9 = torch.index_select(
                self.entity_embedding9,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            h10 = torch.index_select(
                self.entity_embedding10,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            t10 = torch.index_select(
                self.entity_embedding10,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            h11 = torch.index_select(
                self.entity_embedding11,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            t11 = torch.index_select(
                self.entity_embedding11,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            h12 = torch.index_select(
                self.entity_embedding12,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            t12 = torch.index_select(
                self.entity_embedding12,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            # self.node_features= self.node_features.cuda()
            # print(self.node_features.is_cuda)
            # print(self.node_features[0].is_cuda)
            # h_d = self.node_features[0][head_part[:, 0]].view(batch_size, -1)#.cuda()
            # print(h_d.is_cuda)
            # t_d = torch.index_select(self.node_features[0], dim=0, index=tail_part.view(-1)).view(batch_size, negative_sample_size)

            h_d = []  # self.zero_f_tensor.clone()#torch.zeros(6)
            t_d = []  # self.zero_f_tensor.clone()#torch.zeros(6)

            for i in range(0, 5):
                h_d_ = self.node_features[i][head_part[:, 0]].view(batch_size, -1)
                t_d_ = torch.index_select(self.node_features[i], dim=0, index=tail_part.view(-1)).view(batch_size,
                                                                                                       negative_sample_size)
                h_d.append(h_d_)
                t_d.append(t_d_)

            #distances_h = distances_h.view(batch_size, -1)
            #distances_t = distances_t.view(batch_size, -1,negative_sample_size)

            head = [h0, h1,  h3, h4, h5, h7, h8, h9, h10, h11, h12, h_d, distances_h]#h2, h6,
            relation = [r0, r1, r2, r3, r4, r5, r6]# r7
            tail = [t0, t1, t3, t4, t5,t7, t8, t9, t10, t11, t12, t_d, distances_t]# t2, t6,

        else:
            raise ValueError('mode %s not supported' % mode)

        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE,
            'MDE': self.MDE
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score

    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def MDE(self, h, r, t, mode):
        # a = h + r - t
        # b = h + t - r
        # c = t + r - h
        # d = h - r * t
        r_f_0 = torch.abs((torch.cos(h[11][0]) - torch.cos(t[11][0])))
        r_f_1 = torch.abs((torch.cos(h[11][1]) - torch.cos(t[11][1])))
        r_f_2 = torch.abs((torch.cos(h[11][2]) - torch.cos(t[11][2])))
        #r_f_4 = torch.abs((torch.cos(h[12]) - torch.cos(t[12])))#torch.abs((torch.cos(h[11][3]) - torch.cos(t[11][3])))
        r_f_5 = torch.abs((torch.cos(h[11][4]) - torch.cos(t[11][4])))
        score_f_c0 = torch.norm((h[6] - t[6]), p=2, dim=2)
        score_f_c1 = torch.norm((h[7] - t[7]), p=2, dim=2)
        score_f_c2 = torch.norm((h[8] - t[8]), p=2, dim=2)
        score_f_c5 = torch.norm((h[9] - t[9]), p=2, dim=2)
        #print(h[9].size(),t[9].size(),torch.cos(h[12]).size(),torch.cos(t[12]).size(), h[11][2].size(),t[11][2].size())
        #a = h[9]- t[9]

        cuda0 = torch.device('cuda:0')
        cuda2 = torch.device('cuda:2')

        h_d_ = torch.cos(h[12].to(cuda2)).view(h[10].size())
        t_d_ = torch.cos(t[12].to(cuda2)).view(t[10].size())
        score_f_c4 = torch.norm(self.Rotation_EulerAngels(h[10], r[6], t[10], mode), p=2, dim=2).to(
            cuda2) - torch.norm((h_d_ - t_d_), p=2, dim=2).to(cuda2)  # torch.norm((a), p=2, dim=2)- h_d_ -t_d_

        #score_f_c4 = score_f_c4# - r_f_4
        score_f_c0 = score_f_c0 - r_f_0
        score_f_c1 = score_f_c1 - r_f_1
        score_f_c2 = score_f_c2 - r_f_2
        score_f_c5 = score_f_c5 - r_f_5
        if mode == 'head-batch':
            a = h[0].to(cuda2) + (r[0].to(cuda2) - t[0].to(cuda2))
            b = h[1].to(cuda2) + (t[1].to(cuda2) - r[1].to(cuda2))
            e = h[3].to(cuda2) + (r[3].to(cuda2) - t[3].to(cuda2))
            f = h[4].to(cuda2) + (t[4].to(cuda2) - r[4].to(cuda2))
        else:
            a = (h[0].to(cuda2) + r[0].to(cuda2)) - t[0].to(cuda2)
            b = (h[1].to(cuda2) + t[1].to(cuda2)) - r[1].to(cuda2)
            e = (h[3].to(cuda2) + r[3].to(cuda2)) - t[3].to(cuda2)
            f = (h[4].to(cuda2) + t[4].to(cuda2)) - r[4].to(cuda2)

        d = self.Rotation_EulerAngels(h[2], r[2], t[2], mode)
        i = self.Rotation_EulerAngels(h[5], r[5], t[5], mode)
        score_a = (torch.norm(a, p=2, dim=2) + torch.norm(e, p=2, dim=2)) / 2.0
        score_b = (torch.norm(b, p=2, dim=2) + torch.norm(f, p=2, dim=2)) / 2.0
        score_d = (torch.norm(d, p=2, dim=2) + torch.norm(i, p=2, dim=2)) / 2.0

        score_all = (self.w[0].to(cuda2) * score_a.to(cuda2) + self.w[1].to(cuda2) * score_b.to(cuda2) + self.w[3].to(
            cuda2) * score_d.to(cuda2) + self.w[  # self.w[2] * score_c
                         4].to(cuda2) * score_f_c0.to(cuda2) + self.w[5].to(cuda2) * score_f_c1.to(cuda2) + self.w[
                         6].to(cuda2) * score_f_c2.to(cuda2) +
                     self.w[7].to(cuda2) * score_f_c4.to(cuda2) + self.w[8].to(cuda2) * score_f_c5.to(
                    cuda2)) / self.args.psi  # self.w[9] *
        score_all = F.tanhshrink(score_all).to(cuda0)
        score = self.gamma.item() - score_all
        return score

    def MDE_(self, h, r, t, mode):
        # a = h + r - t
        # b = h + t - r
        # c = t + r - h
        # d = h - r * t
        r_f_0 = torch.abs((torch.cos(h[11][0]) - torch.cos(t[11][0])))
        r_f_1 = torch.abs((torch.cos(h[11][1]) - torch.cos(t[11][1])))
        r_f_2 = torch.abs((torch.cos(h[11][2]) - torch.cos(t[11][2])))
        #r_f_4 = torch.abs((torch.cos(h[12]) - torch.cos(t[12])))#torch.abs((torch.cos(h[11][3]) - torch.cos(t[11][3])))
        r_f_5 = torch.abs((torch.cos(h[11][4]) - torch.cos(t[11][4])))
        score_f_c0 = torch.norm((h[6] - t[6]), p=2, dim=2)
        score_f_c1 = torch.norm((h[7] - t[7]), p=2, dim=2)
        score_f_c2 = torch.norm((h[8] - t[8]), p=2, dim=2)
        score_f_c5 = torch.norm((h[9] - t[9]), p=2, dim=2)
        #print(h[9].size(),t[9].size(),torch.cos(h[12]).size(),torch.cos(t[12]).size(), h[11][2].size(),t[11][2].size())
        #a = h[9]- t[9]
        h_d_ = torch.cos(h[12]).view(h[10].size())
        t_d_ = torch.cos(t[12]).view(t[10].size())
        score_f_c4 = self.Rotation_EulerAngels(h[10] ,r[6], t[10], mode) #torch.norm((a), p=2, dim=2)- h_d_ -t_d_
        score_f_c4 = torch.norm(score_f_c4, p=2, dim=2) - torch.norm((h_d_ -t_d_), p=2, dim=2)
        #score_f_c4 = score_f_c4# - r_f_4
        score_f_c0 = score_f_c0 - r_f_0
        score_f_c1 = score_f_c1 - r_f_1
        score_f_c2 = score_f_c2 - r_f_2
        score_f_c5 = score_f_c5 - r_f_5
        if mode == 'head-batch':
            a = h[0] + (r[0] - t[0])
            b = h[1] + (t[1] - r[1])
            e = h[3] + (r[3] - t[3])
            f = h[4] + (t[4] - r[4])
        else:
            a = (h[0] + r[0]) - t[0]
            b = (h[1] + t[1]) - r[1]
            e = (h[3] + r[3]) - t[3]
            f = (h[4] + t[4]) - r[4]

        d = self.Rotation_EulerAngels(h[2], r[2], t[2], mode)
        i = self.Rotation_EulerAngels(h[5], r[5], t[5], mode)
        score_a = (torch.norm(a, p=2, dim=2) + torch.norm(e, p=2, dim=2)) / 2.0
        score_b = (torch.norm(b, p=2, dim=2) + torch.norm(f, p=2, dim=2)) / 2.0
        score_d = (torch.norm(d, p=2, dim=2) + torch.norm(i, p=2, dim=2)) / 2.0

        score_all = (self.w[0] * score_a + self.w[1] * score_b  + self.w[3] * score_d + self.w[ #self.w[2] * score_c
            4] * score_f_c0 + self.w[5] * score_f_c1 + self.w[6] * score_f_c2 + self.w[7] * score_f_c4 + self.w[8] * score_f_c5) / self.args.psi #self.w[9] *
        score_all = F.tanhshrink(score_all)
        score = self.gamma.item() - score_all
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim=2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim=2)
        return score

    def RotatE_score(self, head, relation, tail, mode):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]
        phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)
        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)
        #score = self.gamma.item() - score.sum(dim=2)
        return score

    def Rotation_EulerAngels(self,head, relation, tail, mode):
        cuda0 = torch.device('cuda:0')
        cuda1 = torch.device('cuda:1')
        cuda2 = torch.device('cuda:2')
        cuda3 = torch.device('cuda:3')
        cuda4 = torch.device('cuda:4')
        #head = head.to(cuda2)
        relation = relation.to(cuda1)
        tail = tail.to(cuda1)
        h_1, h_2, h_3 = torch.chunk(head, 3, dim=2)
        t_1, t_2, t_3 = torch.chunk(tail, 3, dim=2)
        h_1 = h_1.to(cuda2)
        h_2 = h_2.to(cuda4)
        h_3 = h_3.to(cuda4)
        relation_1, relation_2, relation_3 = torch.chunk(relation, 3, dim=2)
        pi = 3.14159265358979323846
        phase_relation1 = relation_1 / (self.embedding_range.item() / pi)  # ψ psi  , entity embedding dim default is 100
        phase_relation2 = relation_2 / (self.embedding_range.item() / pi)  # θ theta
        phase_relation3 = relation_3 / (self.embedding_range.item() / pi)  # φ phi
        cos_psi = torch.cos(phase_relation1)
        cos_theta = torch.cos(phase_relation2)
        cos_phi = torch.cos(phase_relation3)
        sin_psi = torch.sin(phase_relation1)
        sin_theta = torch.sin(phase_relation2)
        sin_phi = torch.sin(phase_relation3)

        cos_psi = cos_psi.to(cuda3)
        cos_theta = cos_theta.to(cuda3)
        cos_phi = cos_phi.to(cuda3)
        sin_psi = sin_psi.to(cuda3)
        sin_theta = sin_theta.to(cuda3)
        sin_phi = sin_phi.to(cuda3)
        a1 = cos_theta * cos_psi
        a2 = - cos_phi * sin_psi + sin_phi * sin_theta * cos_psi
        a3 = sin_phi * sin_psi + cos_phi * sin_theta * cos_psi
        b1 = cos_theta * sin_psi
        b2 = cos_phi * cos_psi + sin_phi * sin_theta * sin_psi
        b3 = -sin_phi * cos_psi + cos_phi * sin_theta * sin_psi
        c1 = -sin_theta
        c2 = sin_phi * cos_theta
        c3 = cos_phi * cos_theta
        a1 = a1.to(cuda1)
        a2 = a2.to(cuda1)
        a3 = a3.to(cuda1)
        b1 = b1.to(cuda1)
        b2 = b2.to(cuda1)
        b3 = b3.to(cuda1)
        c1 = c1.to(cuda1)
        c2 = c2.to(cuda1)
        c3 = c3.to(cuda1)
        t_1 = t_1.to(cuda1)
        t_2 = t_2.to(cuda1)
        t_3 = t_3.to(cuda1)
        X = t_1 * a1 + t_2 * a2 + t_3 * a3
        Y = t_1 * b1 + t_2 * b2 + t_3 * b3
        Z = t_1 * c1 + t_2 * c2 + t_3 * c3
        x_score = h_1 - X.to(cuda2)
        x_score = x_score.to(cuda4)

        y_score = h_2 - Y.to(cuda4)
        z_score = h_3 - Z.to(cuda4)
        score = torch.stack([x_score, y_score, z_score], dim=0)
        score = score.norm(dim=0)
        #score = score.to(cuda0)  # cpu()
        return score

    def Rotation_EulerAngels_(self,  head, relation, tail, mode):
        h_1, h_2, h_3 = torch.chunk(head, 3, dim=2)
        t_1, t_2, t_3 = torch.chunk(tail, 3, dim=2)
        relation_1, relation_2, relation_3 = torch.chunk(relation,3,dim=2)

        pi = 3.14159265358979323846
        phase_relation1 = relation_1 / (self.embedding_range.item() / pi)  # ψ psi
        phase_relation2 = relation_2 / (self.embedding_range.item() / pi) # θ theta
        phase_relation3 = relation_3 / (self.embedding_range.item() / pi) # φ phi
        cos_psi = torch.cos(phase_relation1)
        cos_theta = torch.cos(phase_relation2)
        cos_phi = torch.cos(phase_relation3)
        sin_psi = torch.sin(phase_relation1)
        sin_theta = torch.sin(phase_relation2)
        sin_phi = torch.sin(phase_relation3)
        #a is first row
        a1 = cos_theta * cos_psi
        a2 = - cos_phi * sin_psi + sin_phi * sin_theta * cos_psi
        a3 = sin_phi * sin_psi + cos_phi * sin_theta * cos_psi
        b1 = cos_theta * sin_psi
        b2 = cos_phi * cos_psi + sin_phi * sin_theta * sin_psi
        b3 = -sin_phi * cos_psi + cos_phi * sin_theta * sin_psi
        c1 = -sin_theta
        c2 = sin_phi * cos_theta
        c3 = cos_phi * cos_theta

        #if mode == 'head-batch':
        X = t_1 * a1 + t_2 * a2 + t_3 * a3
        Y = t_1 * b1 + t_2 * b2 + t_3 * b3
        Z = t_1 * c1 + t_2 * c2 + t_3 * c3
        x_score = h_1 - X
        y_score = h_2 - Y
        z_score = h_3 - Z
        #else:
        # X = h_1 * a1 + h_2 * a2 + h_3 * a3
        # Y = h_1 * b1 + h_2 * b2 + h_3 * b3
        # Z = h_1 * c1 + h_2 * c2 + h_3 * c3
        # x_score = X - t_1
        # y_score = Y - t_2
        # z_score = Z - t_3
        score = torch.stack([x_score, y_score,z_score], dim=0)
        score = score.norm(dim=0)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=2)
        return score

    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846

        # Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head / (self.embedding_range.item() / pi)
        phase_relation = relation / (self.embedding_range.item() / pi)
        phase_tail = tail / (self.embedding_range.item() / pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim=2) * self.modulus
        return score

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

            y = Variable((torch.Tensor([-1]))).cuda()
            lambda_pos = Variable(torch.FloatTensor([args.gamma_1])).cuda()
            lambda_neg = Variable(torch.FloatTensor([args.gamma_2])).cuda()
        else:
            y = Variable((torch.Tensor([-1])))
            lambda_pos = Variable(torch.Tensor([args.gamma_1]))
            lambda_neg = Variable(torch.Tensor([args.gamma_2]))
        beta_1 = args.beta_1
        beta_2 = args.beta_2

        if args.mde_score:
            negative_score = - model((positive_sample, negative_sample), mode=mode)

            if args.negative_adversarial_sampling:
                negative_score = (negative_score.sum(dim=1) - (args.negative_sample_size * negative_sample.shape[
                    0])) * args.adversarial_temperature  # - (args.negative_sample_size * args.gamma_2)
            else:
                negative_score = negative_score.mean(dim=1)

            positive_score = - model(positive_sample)
            if args.uni_weight:
                positive_sample_loss = positive_score.mean()
                negative_sample_loss = negative_score.mean()
            else:
                positive_sample_loss = (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
                negative_sample_loss = (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

            positive_sample_loss = positive_sample_loss.unsqueeze(dim=0)
            negative_sample_loss = negative_sample_loss.unsqueeze(dim=0)
            loss, positive_sample_loss, negative_sample_loss = model.mde_loss_func(positive_sample_loss,
                                                                                   negative_sample_loss, y, lambda_pos,
                                                                                   lambda_neg, beta_1, beta_2)
            # loss = (positive_sample_loss + negative_sample_loss) / 2

            if args.regularization != 0.0:
                # Use L3 regularization for ComplEx and DistMult
                regularization = args.regularization * (
                        model.entity_embedding.norm(p=3) ** 3 +
                        model.relation_embedding.norm(p=3).norm(p=3) ** 3
                )
                loss = loss + regularization
                regularization_log = {'regularization': regularization.item()}
            else:
                regularization_log = {}

            loss.backward()

            optimizer.step()

            log = {
                **regularization_log,
                'positive_sample_loss': positive_sample_loss.item(),
                'negative_sample_loss': negative_sample_loss.item(),
                'loss': loss.item()
            }

            return log

        negative_score = model((positive_sample, negative_sample), mode=mode)

        if args.negative_adversarial_sampling:
            # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                              * F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)

        positive_score = model(positive_sample)

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        if args.regularization != 0.0:
            # Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                    model.entity_embedding.norm(p=3) ** 3 +
                    model.relation_embedding.norm(p=3).norm(p=3) ** 3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log

    @staticmethod
    def mde_loss_func(p_score, n_score, y, lambda_pos, lambda_neg, beta_1, beta_2):
        criterion = nn.MarginRankingLoss(1.0, False)

        pos_loss = criterion(p_score, lambda_pos, y)
        neg_loss = criterion(n_score, lambda_neg, -y)
        loss = beta_1 * pos_loss + beta_2 * neg_loss
        return loss, pos_loss, neg_loss

    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''

        model.eval()

        if args.countries:
            # Countries S* datasets are evaluated on AUC-PR
            # Process test data for AUC-PR evaluation
            sample = list()
            y_true = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                if args.model == "MDE":
                    y_score = model(sample).squeeze(1).cpu().numpy()
                else:
                    y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            # average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}

        else:
            # Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            # Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples,
                    all_true_triples,
                    args.nentity,
                    args.nrelation,
                    'head-batch'
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num // 2),
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples,
                    all_true_triples,
                    args.nentity,
                    args.nrelation,
                    'tail-batch'
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num // 2),
                collate_fn=TestDataset.collate_fn
            )

            test_dataset_list = [test_dataloader_head, test_dataloader_tail]

            logs = []

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        batch_size = positive_sample.size(0)
                        if args.model == "MDE":
                            score = model((positive_sample, negative_sample), mode)
                            # print("printing scores in test:",score)
                        else:
                            score = model((positive_sample, negative_sample), mode)
                        score += filter_bias

                        # Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim=1, descending=True)

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            # Notice that argsort is not ranking
                            # print("positive arg i",positive_arg[i])
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            # print("ranking:", ranking)
                            assert ranking.size(0) == 1

                            # ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            logs.append({
                                'MRR': 1.0 / ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })

                        if step % args.test_log_steps == 0:
                            # print("printing scores in test:",score)
                            # print("ranking:", ranking)
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

        return metrics