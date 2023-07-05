import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from eval_metrics import *
from model.DynamicGRU import DynamicGRU
from model.bert_modules.bert import BERT

class kerl(nn.Module):
    def __init__(self, num_users, num_items, model_args, device,kg_map):
        super(kerl, self).__init__()

        self.args = model_args
        self.device = device
        self.lamda = 17


        L = self.args.L
        dims = self.args.d
        predict_T=self.args.T

        self.kg_map =kg_map

        self.item_embeddings = nn.Embedding(num_items, dims).to(device)
        self.DP = nn.Dropout(0.5)
        self.enc = DynamicGRU(input_dim=dims,
                             output_dim=dims, bidirectional=False, batch_first=True)

        vocab_size = num_items + 1


        self.enc1 = BERT(self.args).to(device)

        # self.conv1 = nn.Conv1d(in_channels=50, out_channels=1, kernel_size=1)

        # self.att1 = nn.Linear(dims , dims)
        # self.att2 = nn.Linear(dims, dims)
        # self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax()



        self.mlp = nn.Linear(dims*2, dims)

        self.fc = nn.Linear(dims, num_items)


        self.BN = nn.BatchNorm1d(50, affine=False)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, batch_sequences, train_len):


        probs = []
        input=self.enc1(batch_sequences)



        out_enc, h = self.enc(input, train_len)


        kg_map = self.BN(self.kg_map)
        kg_map =kg_map.detach()
        batch_kg = self.get_kg(batch_sequences,train_len,kg_map)






        mlp_in = torch.cat([h.squeeze(), batch_kg], dim=1)



        mlp_hidden = self.mlp(mlp_in)
        mlp_hidden = torch.tanh(mlp_hidden)

        out = self.fc(mlp_hidden)
        probs.append(out)
        return torch.stack(probs, dim=1)

    def RLtrain(self, batch_sequences, items_to_predict, pred_one_hot, train_len,tarlen):
        probs = []
        probs_orgin = []
        each_sample = [] 
        Rewards = []



        input= self.enc1(batch_sequences)

        out_enc, h = self.enc(input,train_len)

        kg_map = self.BN(self.kg_map)
        batch_kg = self.get_kg(batch_sequences,train_len,kg_map)



        mlp_in = torch.cat([h.squeeze(), batch_kg], dim=1)


        mlp_hidden = self.mlp(mlp_in)
        mlp_hidden = torch.tanh(mlp_hidden)
        out_fc = self.fc(mlp_hidden)

        '''
        When sampling episodes, we increased the probability of ground truth to improve the convergence efficiency
        '''
        out_distribution = F.softmax(out_fc, dim=1)
        probs_orgin.append(out_distribution)
        out_distribution = 0.8 * out_distribution
        out_distribution = torch.add(out_distribution, pred_one_hot)
        # pai-->p(a|s)
        probs.append(out_distribution)
        m = torch.distributions.categorical.Categorical(out_distribution)
        # action
        sample1 = m.sample()
        each_sample.append(sample1)


        Reward= self.generateReward(sample1, self.args.T - 1, 1, items_to_predict, pred_one_hot, h,
                                                batch_kg, kg_map, tarlen)

        Rewards.append(Reward)


        probs = torch.stack(probs, dim=1)
        probs_orgin = torch.stack(probs_orgin, dim=1)
        return probs, probs_orgin, torch.stack(each_sample, dim=1), torch.stack(Rewards, dim=1)




    def get_kg(self,batch_sequences,trainlen,kg_map):

        batch_kg = []
        for i, seq in enumerate(batch_sequences):

            seq_kg = kg_map[seq]



            seq_kg_avg = torch.sum(seq_kg,dim=0)
            seq_kg_avg = torch.div(seq_kg_avg,trainlen[i])
            batch_kg.append(seq_kg_avg)







        batch_kg = torch.stack(batch_kg)

        return batch_kg

    def generateReward(self, sample1, path_len, path_num, items_to_predict, pred_one_hot,h_orin,batch_kg,kg_map,tarlen):

        Reward = []
        dist = []

        for paths in range(path_num):
            h = h_orin
            indexes = []
            indexes.append(sample1)
            dec_inp_index = sample1




            dec_inp = self.item_embeddings(dec_inp_index)


            dec_inp = dec_inp.unsqueeze(1)
            ground_kg = self.get_kg(items_to_predict[:, self.args.T - path_len - 1:],tarlen,kg_map)

            for i in range(path_len):



                out_enc, h = self.enc(dec_inp, h, one=True)


                mlp_in = torch.cat([h.squeeze(), batch_kg], dim=1)


                mlp_hidden = self.mlp(mlp_in)
                mlp_hidden = torch.tanh(mlp_hidden)
                out_fc = self.fc(mlp_hidden)

                out_distribution = F.softmax(out_fc, dim=1)
                out_distribution = 0.8 * out_distribution
                out_distribution = torch.add(out_distribution, pred_one_hot)

                m = torch.distributions.categorical.Categorical(out_distribution)
                sample2 = m.sample()

                dec_inp = self.item_embeddings(sample2)

                dec_inp = dec_inp.unsqueeze(1)
                indexes.append(sample2)

            indexes = torch.stack(indexes, dim=1)
            episode_kg = self.get_kg(indexes,torch.Tensor([path_len+1]*len(indexes)),kg_map)

            dist.append(self.cos(episode_kg ,ground_kg))

            Reward.append(dcg_k(items_to_predict[:, self.args.T - path_len - 1:], indexes, path_len + 1))
        Reward = torch.FloatTensor(Reward).to(self.device)

        dist = torch.stack(dist, dim=0)
        dist = torch.mean(dist, dim=0)



        Reward = torch.mean(Reward, dim=0)


        Reward = Reward + self.lamda * dist

        return Reward


    def compare_kgReawrd(self, reward, dist):

        logit_reward, indice = reward.sort(dim=0)

        dist_sort = dist.gather(dim=0, index=indice)
        return dist_sort

