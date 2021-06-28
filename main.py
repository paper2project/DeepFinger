'''
This script handling the training process.
'''

import argparse
import math
import time
import os
from collections import Counter


from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import CNN.Constants as Constants
from dataset import TranslationDataset, paired_collate_fn
from CNN.model import Net
import clustering
from util import UnifLabelSampler

import numpy as np
from data_processing import DataManager
from sklearn.metrics import balanced_accuracy_score

def train_epoch(model, training_data, optimizer, device):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss = 0
    n_total = 0
    n_correct = 0

    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):

        src_seq,src_lbl = map(lambda x: x.to(device), batch)
        optimizer.zero_grad()
        pred = model(src_seq)
        sample = (torch.max(pred, 1)[1] == src_lbl).sum().item()
        n_correct += sample
        n_total += src_lbl.shape[0]

        loss = F.cross_entropy(pred, src_lbl)
        loss.backward()


        optimizer.step()
        # note keeping
        total_loss += loss.item()

    train_acc = 100. * n_correct / n_total
    print("training acc is: ", train_acc, "training loss is: ", total_loss)
    return total_loss, train_acc

def eval_epoch(model, validation_data, device):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    y_true = torch.LongTensor(0).to(device)
    y_predict = torch.LongTensor(0).to(device)

    with torch.no_grad():
        for batch in tqdm(
                validation_data, mininterval=2,
                desc='  - (Validation) ', leave=False):

            src_seq,src_lbl= [x.to(device) for x in batch]
            pred = model(src_seq)

            y_predict = torch.cat([y_predict, torch.max(pred, 1)[1]], 0)
            y_true = torch.cat([y_true, src_lbl], 0)

            loss = F.cross_entropy(pred, src_lbl)
            total_loss += loss.item()

    y_true = y_true.cpu().numpy().tolist()
    y_predict = y_predict.cpu().numpy().tolist()
    y_true_trans = np.array(y_true)
    y_predict_trans = np.array(y_predict)

    acc = balanced_accuracy_score(y_true_trans, y_predict_trans)
    valid_acc = 100. * acc
    print("validation acc is: ", valid_acc, "validation loss is: ", total_loss)
    return y_true,y_predict,total_loss, valid_acc

def train(model, training_data, validation_data, testing_data, optimizer, device, opt):
    ''' Start training '''

    log_train_file = None
    log_valid_file = None

    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + '.valid.log'
        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    valid_accus = []
    test_accus=[]


    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(
            model, training_data, optimizer, device)
        print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  ppl=math.exp(min(train_loss, 100)), accu=train_accu,
                  elapse=(time.time()-start)/60))


        start = time.time()
        _,_,valid_loss, valid_accu = eval_epoch(model, validation_data, device)
        print('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
                'elapse: {elapse:3.3f} min'.format(
                    ppl=math.exp(min(valid_loss, 100)), accu=valid_accu,
                    elapse=(time.time()-start)/60))
        valid_accus += [valid_accu]


        start = time.time()
        y_true,y_predict,test_loss, test_accu = eval_epoch(model, testing_data, device)
        print('  - (Testing) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
                'elapse: {elapse:3.3f} min'.format(
                    ppl=math.exp(min(test_loss, 100)), accu=test_accu,
                    elapse=(time.time()-start)/60))
        test_accus += [test_accu]

        ## save test-pred-labels
        with open(os.path.join(opt.save_pred_result,'label_test_pred'+str(opt.fold_num)+'_'+str(epoch_i)+'.txt'),'w') as f:
            for i in y_true:
                f.write(str(i)+' ')
            f.write('\n')
            for j in y_predict:
                f.write(str(j)+' ')

        ## save model
        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i,
            'valid_accu': valid_accu,
            'optimizer': optimizer}

        if opt.save_model:
            if opt.save_model_mode == 'all':
                model_name = opt.save_model + \
                             '_fold_' + str(opt.fold_num) + \
                             '_epoch_' + str(epoch_i) + \
                             '_fs' + str(opt.feature_size) + \
                             '_ks' + str(opt.kernel_size) + \
                             '_es' + str(opt.embedding_size) + \
                             '_sl' + str(opt.max_word_seq_len) + \
                             '_dp' + str(opt.dropout) + \
                             '_accu_{accu:3.3f}'.format(accu=100 * valid_accu) + \
                             '.chkpt'
                torch.save(checkpoint, model_name)
            elif opt.save_model_mode == 'step':
                if (epoch_i+1)%opt.save_model_epoch_step==0:
                    model_name = opt.save_model + \
                                 '_fold_' + str(opt.fold_num) + \
                                 '_epoch_' + str(epoch_i) + \
                                 '_fs' + str(opt.feature_size) + \
                                 '_ks' + str(opt.kernel_size) + \
                                 '_es' + str(opt.embedding_size) + \
                                 '_sl' + str(opt.max_word_seq_len) + \
                                 '_dp' + str(opt.dropout) + \
                                 '_accu_{accu:3.3f}'.format(accu=100 * valid_accu) + \
                                 '.chkpt'
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

        ## save log
        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))

def main(opt):
    ## opt operate
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.CUDA_VISIBLE_DEVICES
    opt.cuda = not opt.no_cuda
    opt.kernel_size_list=[int(x) for x in opt.kernel_size.split(',')]

    result_dir='_fs'+str(opt.feature_size)+\
               '_ks'+str(opt.kernel_size)+\
               '_es'+str(opt.embedding_size)+\
               '_sl'+str(opt.max_word_seq_len)+\
               '_dp'+str(opt.dropout)+\
               '_cn'+str(opt.cluster_num)+\
               '_eps'+str(opt.eps)+ \
               '_ms' + str(opt.min_samples) + \
               '_fn' + str(opt.fold_num) + \
               '/'
    opt.save_model=opt.save_model+opt.train_src.split('/')[-1]+'/'+result_dir
    os.makedirs(opt.save_model,exist_ok=True)
    opt.save_pred_result=opt.save_pred_result+opt.train_src.split('/')[-1]+'/'+result_dir
    os.makedirs(opt.save_pred_result, exist_ok=True)
    print(opt.save_pred_result)

    for label_choose in range(opt.grained):
        #========= Processing Dataset =========#
        if(opt.data==''):
            datamanager = DataManager(opt.train_src,opt.grained,label_choose,opt.max_word_seq_len,
                                      opt.fold_num,opt.min_word_count,opt.save_preprocess_data)
            data=datamanager.getdata()
        # #========= Loading Dataset =========#
        else:
            data = torch.load(opt.data)


        training_data, validation_data,testing_data = prepare_dataloaders(data, opt)
        opt.src_vocab_size = training_data.dataset.src_vocab_size
        print(opt)

        # clustering algorithm to use
        deepcluster = clustering.__dict__['Kmeans'](opt.cluster_num,opt.eps,opt.min_samples)

        device = torch.device('cuda' if opt.cuda else 'cpu')
        CNNnet=Net(opt.src_vocab_size,
                   opt.max_word_seq_len,
                   opt.embedding_size,


                   opt.feature_size,
                   opt.kernel_size_list,
                   opt.grained).to(device)

        fd = int(CNNnet.top_layer.weight.size()[1])
        print('fd:',fd)

        latest_cluster_index={}
        for iteration_i in range(opt.iteration):

            print('[ iteration', iteration_i, ']')
            CNNnet.top_layer = None
            features = compute_features(training_data, CNNnet, len(data['train']['src']))

            deepcluster.cluster(features,data['train']['src'], verbose=True)

            train_dataset_cluster,cluster_label_list= clustering.cluster_assign(deepcluster.cluster_label_list,
                                                  data['train']['src'],iteration_i,latest_cluster_index)

            with open(os.path.join(opt.save_pred_result,'{}_cluster_data.txt'.format(label_choose)),'w') as f:
                for i,data_x in enumerate(data['origin']['train']):

                    f.write(str(cluster_label_list[i])+'\t'+str(data_x)+'\n')

            latest_cluster_index=cluster_label_list

            train_dataloader_cluster = prepare_dataloaders_cluster(train_dataset_cluster, opt, data)

            # set last fully connected layer
            CNNnet.top_layer = nn.Linear(fd, len(set(deepcluster.cluster_label_list)))
            CNNnet.top_layer.weight.data.normal_(0, 0.01)
            CNNnet.top_layer.bias.data.zero_()
            CNNnet.top_layer.cuda()

            learnrate = 5*1e-5
            optimizer = optim.Adam(
                    filter(lambda x: x.requires_grad, CNNnet.parameters()), lr=learnrate,
                    betas=(0.9, 0.98), eps=1e-09)

            # train(CNNnet, train_dataloader_cluster, optimizer, device, opt)
            for i in range(opt.epoch):
                train_loss, train_accu = train_epoch(
                    CNNnet, train_dataloader_cluster, optimizer, device)
                print(label_choose,train_loss,train_accu)



def compute_features(dataloader, model, N):

    model.eval()
    # discard the label information in the dataloader级
    for i, (input_tensor, _) in enumerate(dataloader):
        input_var = torch.autograd.Variable(input_tensor.cuda(), volatile=True)
        aux = model(input_var).data.cpu().numpy()

        if i == 0:
            features = np.zeros((N, aux.shape[1]), dtype='float32')
        aux = aux.astype('float32')
        if i < len(dataloader) - 1:
            features[i * opt.batch_size: (i + 1) *opt.batch_size] = aux
        else:
            # special treatment for final batch
            features[i * opt.batch_size:] = aux
    print(features.shape)

    return features

def prepare_dataloaders(data, opt):
    # ========= Preparing DataLoader =========#
    train_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=data['dict']['src'],
            src_insts=data['train']['src'],
            src_lbls=data['train']['lbl']),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn)

    valid_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=data['dict']['src'],
            src_insts=data['valid']['src'],
            src_lbls=data['valid']['lbl']),

        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn)

    test_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=data['dict']['src'],
            src_insts=data['test']['src'],
            src_lbls=data['test']['lbl']),

        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn)

    return train_loader, valid_loader, test_loader

def prepare_dataloaders_cluster(train_dataset_cluster,opt,data):

    data_src=[data[0] for data in train_dataset_cluster]
    data_lbl = [data[1] for data in train_dataset_cluster]
    print('data_lbl',Counter(data_lbl))


    train_dataloader_cluster = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=data['dict']['src'],
            src_insts=data_src,
            src_lbls=data_lbl),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn)
    return train_dataloader_cluster


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-cluster_num', type=int, default=30)
    parser.add_argument('-eps', type=float, default=0.1)
    parser.add_argument('-min_samples', type=int, default=10)

    parser.add_argument('-grained', type=int, default=8)
    parser.add_argument('-train_src', default='')
    parser.add_argument('-save_preprocess_data', default='')
    parser.add_argument('-max_word_seq_len', type=int, default=32)
    parser.add_argument('-min_word_count', type=int, default=0)
    parser.add_argument('-fold_num', type=int, default=5)

    parser.add_argument('-epoch', type=int, default=50)
    parser.add_argument('-iteration', type=int, default=5)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-embedding_size', type=int, default=128)
    parser.add_argument('-feature_size', type=int, default=128)
    parser.add_argument('-kernel_size', type=str, default='2,3,4')
    parser.add_argument('-dropout', type=float, default=0.4)

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-CUDA_VISIBLE_DEVICES', type=str, default='1')

    parser.add_argument('-data', type=str, default='')
    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default='model/')
    parser.add_argument('-save_model_mode', type=str, choices=['all', 'step'], default='step')
    parser.add_argument('-save_model_epoch_step', type=int, default=5)
    parser.add_argument('-save_pred_result', default='cluster_data/')


    parser.add_argument('--reassign', type=float, default=1.,
                        help="""how many epochs of training between two consecutive
                        reassignments of clusters (default: 1)""")
    opt = parser.parse_args()
    strat=time.time()
    main(opt)
    print(time.time()-strat)
