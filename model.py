import torch
import torch.nn as nn
import numpy as np

# make a cnn
class Net(torch.nn.Module):
    def __init__(self,vocab_size,max_len,embedding_size,feature_size,kernel_size_list,class_num):
        super(Net, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_size)

        self.conv1 = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv1d(in_channels=embedding_size,out_channels=feature_size,kernel_size=h),
                torch.nn.ReLU(),
                torch.nn.MaxPool1d(kernel_size=max_len-h+1)
            )
            for h in kernel_size_list]
        )

        self.top_layer= torch.nn.Linear(feature_size*len(kernel_size_list), class_num)

    def forward(self, x):
        embed_x = self.embedding(x) # bsize*maxlen*word_dim

        x = embed_x.permute(0, 2, 1)

        out_conv1= [conv(x) for conv in self.conv1]
        out_conv1_cat= torch.cat(out_conv1, dim=1)
        out_conv1_cat = out_conv1_cat.view(-1, out_conv1_cat.size(1))

        if(self.top_layer):
            out_fc1 = self.top_layer(out_conv1_cat)
        else:
            out_fc1=out_conv1_cat

        return out_fc1