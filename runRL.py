from interactions import Interactions
from eval_metrics import *
import pandas as pd

import pdb
import argparse
import logging
from time import time
import datetime
import torch
from model.KERL import kerl
import torch.nn.functional as F
import random
import pickle
logging.basicConfig(level=logging.INFO,filename='train.log',filemode = 'w')
#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

def generate_testsample(test_set,itemnum):
    '''
    input
        test_set: ground-truth
        itemnum: item number
    output
        all_sample:randomly sampled 100 negative items and 1 positive items
    '''

    all_sample =[]

    for eachset in test_set:
        testsample = []
        for i in range(1):
            onesample = []
            onesample +=[eachset[i]]
            other = list(range(1, itemnum))
            other.remove(eachset[i])
            neg = random.sample(other,100)
            onesample +=neg
            testsample.append(onesample)
        testsample = np.stack(testsample)
        all_sample.append(testsample)
    all_sample = np.stack(all_sample)
    return all_sample


def evaluation_kerl(kerl, train, test_set):
    num_users = train.num_users
    num_items = train.num_items
    batch_size = 1024

    user_indexes = np.arange(num_users)
    item_indexes = np.arange(num_items)
    pred_list = None
    test_sequences = train.test_sequences.sequences
    test_len = train.test_sequences.length
    train_idxs = np.arange(test_sequences.shape[0])
    num_batches = int(len(train_idxs) / batch_size) + 1
    test_user=train.test_users
    test1=[]
    for i in range(len(test_user)):
        test=test_set[test_user[i]]
        test1.append(test)

    all_sample = generate_testsample(test1,num_items)
    for batchID in range(num_batches):
        start = batchID * batch_size
        end = start + batch_size

        if batchID == num_batches - 1:
            if start < len(train_idxs):
                end = len(train_idxs)
            else:
                break

        batch_user_index = user_indexes[start:end]


        batch_test_sequences = test_sequences[batch_user_index]

        batch_test_sequences = np.atleast_2d(batch_test_sequences)
        batch_test_len = test_len[batch_user_index]

        batch_test_len = torch.from_numpy(batch_test_len).type(torch.LongTensor).to(device)
        batch_test_sequences = torch.from_numpy(batch_test_sequences).type(torch.LongTensor).to(device)

        prediction_score = kerl(batch_test_sequences, batch_test_len)
        rating_pred = prediction_score
        rating_pred = rating_pred.cpu().data.numpy().copy()

        if batchID == 0:
            pred_list = rating_pred
        else:
            pred_list = np.append(pred_list, rating_pred, axis=0)



    all_top10 = []

    for i in range(1):
        oneloc_top10 = []
        user_index = 0

        for each_policy, each_s in zip(pred_list[:, i, :], all_sample):

            each_s=each_s[i,:]
            each_sample = -each_policy[each_s]
            top10index = np.argsort(each_sample)[:10]
            top10item = each_s[top10index]
            oneloc_top10.append(top10item)
        oneloc_top10=np.stack(oneloc_top10)
        all_top10.append(oneloc_top10)
        user_index +=1
    all_top10 = np.stack(all_top10,axis=1)
    pred_list = all_top10

    precision, ndcg = [], []
    k=10
    for i in range(1):
        pred = pred_list[:,i,:]
        precision.append(precision_at_k(test1, pred, k,i))
        ndcg.append(ndcg_k(test1, pred, k,i))


    return precision, ndcg

def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):

    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop

def train_kerl(train_data, test_data, config,kg_map):
    num_users = train_data.num_users
    num_items = train_data.num_items


    sequences_np = train_data.sequences.sequences

    targets_np = train_data.sequences.targets

    users_np = train_data.sequences.user_ids
    trainlen_np = train_data.sequences.length

    tarlen_np = train_data.sequences.tarlen

    n_train = sequences_np.shape[0]

    logger.info("Total training records:{}".format(n_train))


    kg_map = torch.from_numpy(kg_map).type(torch.FloatTensor).to(device)
    kg_map.requires_grad=False
    seq_model = kerl(num_users, num_items, config, device, kg_map).to(device)


    optimizer = torch.optim.Adam(seq_model.parameters(), lr=config.learning_rate,weight_decay=config.l2)

    lamda = 0.5
    print("loss lamda=",lamda)
    CEloss = torch.nn.CrossEntropyLoss()

    margin = 0.0
    MRLoss = torch.nn.MarginRankingLoss(margin=margin)

    record_indexes = np.arange(n_train)
    batch_size = config.batch_size
    num_batches = int(n_train / batch_size) + 1

    stopping_step = 0
    cur_best_pre_0 = 0
    should_stop = False
    enloss=[]
    time1=[]
    time2=[]
    H10=[]
    NDCG=[]
    for epoch_num in range(config.n_iter):
        t1 = time()
        loss=0

        seq_model.train()

        np.random.shuffle(record_indexes)
        epoch_reward=0.0
        epoch_loss = 0.0
        for batchID in range(num_batches):
            start = batchID * batch_size
            end = start + batch_size

            if batchID == num_batches - 1:
                if start < n_train:
                    end = n_train
                else:
                    break

            batch_record_index = record_indexes[start:end]

            batch_users = users_np[batch_record_index]
            batch_sequences = sequences_np[batch_record_index]
            batch_targets = targets_np[batch_record_index]
            trainlen = trainlen_np[batch_record_index]

            tarlen = tarlen_np[batch_record_index]

            tarlen = torch.from_numpy(tarlen).type(torch.LongTensor).to(device)
            trainlen = torch.from_numpy(trainlen).type(torch.LongTensor).to(device)
            batch_users = torch.from_numpy(batch_users).type(torch.LongTensor).to(device)
            batch_sequences = torch.from_numpy(batch_sequences).type(torch.LongTensor).to(device)
            batch_targets = torch.from_numpy(batch_targets).type(torch.LongTensor).to(device)

            items_to_predict = batch_targets


            if epoch_num>=0:
                pred_one_hot = np.zeros((len(batch_users),num_items))
                batch_tar=targets_np[batch_record_index]
                for i,tar in enumerate(batch_tar):

                    pred_one_hot[i][tar]=0.2/config.T

                pred_one_hot = torch.from_numpy(pred_one_hot).type(torch.FloatTensor).to(device)

                prediction_score, orgin, batch_targets, Reward= seq_model.RLtrain(batch_sequences,items_to_predict,
                                                                                              pred_one_hot, trainlen,
                                                                                              tarlen)



                target = torch.ones((len(prediction_score))).to(device)


                orgin = orgin.view(prediction_score.shape[0] * prediction_score.shape[1], -1)
                target = batch_targets.view(batch_targets.shape[0]*batch_targets.shape[1])
                reward = Reward.view(Reward.shape[0]*Reward.shape[1]).to(device)

                celoss=CEloss(orgin,target)

                prob = torch.index_select(orgin,1,target)
                prob = torch.diagonal(prob,0)
                RLloss =-torch.mean(torch.mul(reward,torch.log(prob)))
                loss=RLloss+celoss


                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


        epoch_loss /= num_batches
        t2 = time()

        enloss.append(epoch_loss)
        time1.append(t2 - t1)

        output_str = "Epoch %d [%.1f s]  loss=%.4f" % (epoch_num + 1, t2 - t1, epoch_loss)

        logger.info(output_str)

        if (epoch_num + 1) > 1:
            seq_model.eval()
            precision, ndcg = evaluation_kerl(seq_model, train_data, test_data)

            H10.append(precision)
            NDCG.append(ndcg)
            time2.append(time() - t2)
            logger.info(', '.join(str(e) for e in precision))
            logger.info(', '.join(str(e) for e in ndcg))
            logger.info("Evaluation time:{}".format(time() - t2))
            cur_best_pre_0, stopping_step, should_stop = early_stopping(precision[0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc',
                                                                    flag_step=5)

            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
            if should_stop == True:
                break
    res = pd.DataFrame(columns=['traintime','enloss'])
    res['traintime'] = time1
    res['enloss'] = enloss

    re1=pd.DataFrame(columns=['h10','ndcg10','evaltime'])
    re1['h10']=H10
    re1['ndcg10']=NDCG
    re1['evaltime']=time2


    res.to_csv('5beauty-jieguo1.csv')
    re1.to_csv('5beauty-jieguo2.csv')
    logger.info("\n")
    logger.info("\n")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data arguments
    #L: max sequence length
    #T: episode length
    parser.add_argument('--L', type=int, default=50)
    parser.add_argument('--T', type=int, default=1)

    parser.add_argument('--model_init_seed', type=int, default=0)
    # BERT #
    parser.add_argument('--bert_max_len', type=int, default=50, help='Length of sequence for bert')
    parser.add_argument('--bert_num_items', type=int, default=12101, help='Number of total items')
    parser.add_argument('--bert_hidden_units', type=int, default=50, help='Size of hidden vectors (d_model)')
    parser.add_argument('--bert_num_blocks', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--bert_num_heads', type=int, default=2, help='Number of heads for multi-attention')
    parser.add_argument('--bert_dropout', type=float, default=0.1,
                        help='Dropout probability to use throughout the model')
    parser.add_argument('--bert_mask_prob', type=float, default=0,
                        help='Probability for masking items in the training sequence')


    # train arguments
    parser.add_argument('--n_iter', type=int, default=100)#100个负样本，训练的epoch_num
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=0)

    # model dependent arguments
    parser.add_argument('--d', type=int, default=50)

    config = parser.parse_args()

    from data import Amazon
    data_set = Amazon.Beauty()  # Books, CDs, LastFM
    train_set, test_set, num_users, num_items,kg_map = data_set.generate_dataset(index_shift=1)
    #pdb.set_trace()


    maxlen = 0


    for inter in train_set:
        if len(inter)>maxlen:
            maxlen=len(inter)

    train = Interactions(train_set, num_users, num_items)




    train.to_newsequence(config.L, config.T)


    logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info(config)

    train_kerl(train,test_set,config,kg_map)
