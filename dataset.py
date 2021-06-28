import numpy as np
import torch
import torch.utils.data

from CNN import Constants

def paired_collate_fn(insts):
    src_insts,src_lbls = list(zip(*insts))
    src_insts=torch.LongTensor(src_insts)
    src_lbls = torch.LongTensor([int(x) for x in src_lbls])
    return (src_insts,src_lbls)


class TranslationDataset(torch.utils.data.Dataset):
    def __init__(
        self,src_word2idx,
        src_insts=None, src_lbls=None):

        assert src_insts

        #revised by yipengw
        src_idx2word = {idx:word for word, idx in src_word2idx.items()}
        self._src_word2idx = src_word2idx
        self._src_idx2word = src_idx2word
        self._src_insts = src_insts

        self._src_lbls = src_lbls


    @property
    def n_insts(self):
        ''' Property for dataset size '''
        return len(self._src_insts)

    @property
    def src_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._src_word2idx)


    @property
    def src_word2idx(self):
        ''' Property for word dictionary '''
        return self._src_word2idx


    @property
    def src_idx2word(self):
        ''' Property for index dictionary '''
        return self._src_idx2word


    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        return self._src_insts[idx],self._src_lbls[idx]
