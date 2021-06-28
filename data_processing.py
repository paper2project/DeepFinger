
import json, random
import re
import os
import torch
import CNN.Constants as Constants
import pickle

class DataManager(object):

    def __init__(self, train_src, grained,label_choose,max_word_seq_len,fold_num,min_word_count,save_data):

        self.train_src=train_src
        self.max_word_seq_len=max_word_seq_len
        self.fold_num=fold_num
        self.min_word_count=min_word_count
        self.save_data=save_data

        ## fold-dataset
        all_file=[(x+2*(fold_num-1))%10 for x in range(10)]
        train_valid_test_file=[all_file[2:],all_file[:2],all_file[2:4]]
        print(train_valid_test_file)

        ## read data
        total_data=[[],[],[],[],[],[]]
        all_data=[[],[],[]]
        index_list=[]
        for i in range(3):
            file=train_valid_test_file[i]
            for num in file:
                file_path=os.path.join(self.train_src,str(num)+'.txt')
                now_all_src_word_insts, now_all_src_lbls,now_all_data=self.read_instances_from_file(file_path,grained,label_choose,
                                                            self.max_word_seq_len)
                total_data[2*i].extend(now_all_src_word_insts)
                total_data[2*i+1].extend(now_all_src_lbls)
                all_data[i].extend(now_all_data)

            cc = list(zip(total_data[2*i], total_data[2*i+1],all_data[i]))
            total_data[2*i], total_data[2*i+1],all_data[i]= zip(*cc)

        [self.train_src_word_insts, self.train_src_lbls,
         self.valid_src_word_insts, self.valid_src_lbls,
         self.test_src_word_insts,  self.test_src_lbls]=total_data
        [self.train_origin_data,self.valid_origin_data,self.test_origin_data]=all_data




    def read_instances_from_file(self,inst_file,grained,label_choose,max_sent_len):
        ''' Convert file into word seq lists and vocab '''
        word_insts = []
        labels = []
        all_data=[]
        trimmed_sent_count = 0

        lable_lis=[x for x in range(grained)] if label_choose=='' else [int(label_choose)]

        with open(inst_file) as f:
            for sent in f:
                [label,payload_str] = sent.strip().split('\t')
                payload=payload_str.split(' ')
                if len(payload) > max_sent_len:
                    trimmed_sent_count += 1
                payload_list = payload[:max_sent_len]

                if ((len(payload_list)!=0) and (int(label) in lable_lis)):
                    labels += [label]
                    now_word_inst=payload_list+[Constants.PAD_WORD]*(max_sent_len-len(payload_list))+\
                                  payload_list[::-1]+[Constants.PAD_WORD]*(max_sent_len-len(payload_list))


                    word_insts += [now_word_inst]
                    all_data.append(payload_str)

        print('[Info] Get {} instances from {}'.format(len(word_insts), inst_file))

        if trimmed_sent_count > 0:
            print('[Warning] {} instances are trimmed to the max sentence length {}.'
                  .format(trimmed_sent_count, max_sent_len))

        return word_insts, labels,all_data

    def build_vocab_idx(self,word_insts, min_word_count,PAD,UNK):
        ''' Trim vocab by number of occurence '''

        full_vocab = set(w for sent in word_insts for w in sent)
        print('[Info] Original Vocabulary size =', len(full_vocab))
        word2idx = {
            PAD: Constants.PAD,
            UNK: Constants.UNK}

        word_count = {w: 0 for w in full_vocab}
        for sent in word_insts:
            for word in sent:
                word_count[word] += 1

        ignored_word_count = 0
        sorted_dict = sorted(word_count.items(), key=lambda d: d[0], reverse=False)

        for (word,count) in sorted_dict:
            if word not in word2idx:
                if count > min_word_count:
                    word2idx[word] = len(word2idx)
                else:
                    ignored_word_count += 1

        print('[Info] Trimmed vocabulary size = {},'.format(len(word2idx)),
              'each with minimum occurrence = {}'.format(min_word_count))
        print("[Info] Ignored word count = {}".format(ignored_word_count))
        return word2idx

    def convert_instance_to_idx_seq(self, word_insts, src_word2idx):
        ''' Mapping words to idx sequence. '''

        payload_data = word_insts
        payload_seq = [[src_word2idx.get(w, Constants.UNK) for w in s] for s in payload_data]

        return payload_seq


    def getdata(self):
        print('[Info] Build vocabulary for source.')
        true_data = self.train_src_word_insts
        zero_line=0
        src_word2idx = self.build_vocab_idx(true_data, self.min_word_count, Constants.PAD_WORD,
                                                 Constants.UNK_WORD)
        print(src_word2idx)

        print('[Info] Convert source word instances into sequences of word index.')

        self.train_src_insts = self.convert_instance_to_idx_seq(self.train_src_word_insts, src_word2idx)
        self.valid_src_insts = self.convert_instance_to_idx_seq(self.valid_src_word_insts,  src_word2idx)
        self.test_src_insts = self.convert_instance_to_idx_seq(self.test_src_word_insts, src_word2idx)

        value_data=[self.train_src_insts, self.valid_src_insts, self.test_src_insts,
                    self.train_src_lbls, self.valid_src_lbls, self.test_src_lbls]

        for data in value_data:
            print('data num:',len(data))
            print(data[:10])


        data = {
            'settings': [self.fold_num,self.max_word_seq_len,self.min_word_count],
            'dict': {
                'src': src_word2idx},
            'train': {
                'src': self.train_src_insts,
                'lbl': self.train_src_lbls},
            'valid': {
                'src': self.valid_src_insts,
                'lbl': self.valid_src_lbls},
            'test': {
                'src': self.test_src_insts,
                'lbl': self.test_src_lbls},
            'origin':{
                'train':self.train_origin_data,
                'valid':self.valid_origin_data,
                'test':self.test_origin_data,
            }
        }

        origin_data='origin_data'
        os.makedirs(origin_data,exist_ok=True)

        with open(os.path.join(origin_data,'train_origin_data.txt'),'w') as f:
            for now_data in self.train_origin_data:
                f.write(str(now_data)+'\n')



        return data


