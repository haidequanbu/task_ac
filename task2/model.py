# two model:RNN and CNN

import torch
import torch.nn as nn
import torch.nn.functional as F


class model_RNN(nn.Module):
    def __init__(self, train_size):
        super(model_RNN, self).__init__()
        self.embbedding = nn.Embedding(train_size, 64)
        self.rnn = nn.LSTM(input_size=64, hidden_size=128, bidirectional=True)
        # self.gru=nn.GRU(input_size=64,hidden_size=128,bidirectional=True)
        self.f1 = nn.Sequential(nn.Linear(256, 5), nn.Softmax())

    def forward(self, x):
        x = self.embbedding(x)
        x = x.permute(1,0,2)
        x,(h_n,c_n)=self.rnn(x)
        final_feature_map=F.dropout(h_n,0.8)
        feature_map=torch.cat([final_feature_map[i,:,:] for i in range(final_feature_map.shape[0])],dim=1)
        final_out=self.f1(feature_map)
        return final_out


class model_CNN(nn.Module):
    def __init__(self,train_size):
        super(model_CNN, self).__init__()
        self.embedding=nn.Embedding(train_size,64)
        self.conv=nn.Sequential(nn.Conv1d(in_channels=64,out_channels=256,kernel_size=5),nn.ReLU(),
                                nn.MaxPool1d(kernel_size=596))
        self.f1=nn.Linear(256,10)

    def forward(self,x):
        x=self.embedding(x)
        x=x.permute(0,2,1)
        x=self.conv(x)
        x=x.view(-1,x.size(1))
        x=F.dropout(x,0.8)
        x=self.f1(x)
        return x



