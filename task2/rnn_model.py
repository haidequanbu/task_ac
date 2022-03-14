import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sklearn
from torchtext.legacy import data

# print(os.getcwd())

dir_all_data = 'data/train.tsv'

# 超参数设置
BATCH_SIZE = 1000
cpu = False  # True   False
if cpu:
    USE_CUDA = False
    DEVICE = torch.device('cpu')
else:
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(0)

# 从文件中读取数据
data_all = pd.read_csv(dir_all_data, sep='\t')
# print(all_data.shape)    #(156060, 4)
# print(all_data.keys())   #['PhraseId', 'SentenceId', 'Phrase', 'Sentiment']
idx = np.arange(data_all.shape[0])
# print(data_all.head())
# print(type(idx))   #<class 'numpy.ndarray'>

# shuffle、划分验证集、测试集,并保存
seed = 0
np.random.seed(seed)
# print(idx)
np.random.shuffle(idx)
# print(idx)

train_size = int(len(idx) * 0.75)
test_size = int(len(idx) * 0.25)

data_all.iloc[idx[:train_size], :].to_csv('data/task2_train.csv', index=False)
data_all.iloc[idx[train_size:test_size], :].to_csv("data/task2_test.csv", index=False)
# 这个有啥用？
data_all.iloc[idx[test_size:], :].to_csv("data/task2_dev.csv", index=False)

# 看不懂，暂时也看不懂
# 使用Torchtext采用声明式方法加载数据
PAD_TOKEN = '<pad>'
TEXT = data.Field(sequential=True, batch_first=True, lower=True, pad_token=PAD_TOKEN)
LABEL = data.Field(sequential=False, batch_first=True, unk_token=None)

# 读取数据
datafields = [("PhraseId", None),  # 不需要的filed设置为None
              ("SentenceId", None),
              ('Phrase', TEXT),
              ('Sentiment', LABEL)]
train_data = data.TabularDataset(path='data/task2_train.csv', format='csv',
                                 fields=datafields)
# dev_data是干嘛用的
dev_data = data.TabularDataset(path='data/task2_dev.csv', format='csv',
                               fields=datafields)
test_data = data.TabularDataset(path='data/task2_test.csv', format='csv',
                                fields=datafields)

# 构建词典，字符映射到embedding
# TEXT.vocab.vectors 就是词向量
TEXT.build_vocab(train_data, vectors='glove.6B.50d',  # 可以提前下载好
                 unk_init=lambda x: torch.nn.init.uniform_(x, a=-0.25, b=0.25))
LABEL.build_vocab(train_data)

# 下面又是干嘛的？
# 得到索引，PAD_TOKEN='<pad>'
PAD_INDEX = TEXT.vocab.stoi[PAD_TOKEN]
TEXT.vocab.vectors[PAD_INDEX] = 0.0

# 构建迭代器
train_iterator = data.BucketIterator(train_data, batch_size=BATCH_SIZE,
                                     train=True, shuffle=True, device=DEVICE)

dev_iterator = data.Iterator(dev_data, batch_size=len(dev_data), train=False,
                             sort=False, device=DEVICE)

test_iterator = data.Iterator(test_data, batch_size=len(test_data), train=False,
                              sort=False, device=DEVICE)

embedding_choice = 'glove'  # 'static'    'non-static'
num_embeddings = len(TEXT.vocab)
embedding_dim = 50
dropout_p = 0.5
hidden_size = 50  # 隐藏单元数
num_layers = 3  # 层数

vocab_size = len(TEXT.vocab)
label_num = len(LABEL.vocab)
print(vocab_size, label_num)


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()

        self.embedding_choice = embedding_choice
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if self.embedding_choice == 'rand':
            self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        if self.embedding_choice == 'glove':
            self.embedding = nn.Embedding(num_embeddings, embedding_dim,
                                          padding_idx=PAD_INDEX).from_pretrained(TEXT.vocab.vectors, freeze=True)
        # input_size (输入的特征维度),hidden_size ,num_layers
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers,
                            batch_first=True, dropout=dropout_p, bidirectional=True)
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(hidden_size * 2, label_num)  # 2 for bidirection

    def forward(self, x):  # (Batch_size, Length)
        # Set initial hidden and cell states
        # h_n (num_layers * num_directions, batch, hidden_size)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
        # c_n (num_layers * num_directions, batch, hidden_size):
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)

        if USE_CUDA:
            h0 = h0.cuda()
            c0 = c0.cuda()

        x = self.embedding(x)  # (Batch_size, Length)
        # (Batch_size,  Length, Dimention)

        out, _ = self.lstm(x, (h0, c0))  # (Batch_size, Length，Dimention)
        # (batch_size, Length, hidden_size)
        out = self.dropout(out)

        out = self.fc(out[:, -1, :])  # (batch_size, Length, hidden_size)
        # (batch_size, label_num)
        return out


model = LSTM()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 创建优化器SGD
criterion = nn.CrossEntropyLoss()  # 损失函数

if USE_CUDA:
    model.cuda()

epoch = 100
best_accuracy = 0.0
start_time = time.time()

for epoch in range(epoch):
    start=time.time()
    model.train()
    total_loss = 0.0
    accuracy = 0.0
    total_correct = 0.0
    total_data_num = len(train_iterator.dataset)
    steps = 0.0
    # 训练
    for batch in train_iterator:
        steps += 1
        # print(steps)
        optimizer.zero_grad()  # 梯度缓存清零

        batch_text = batch.Phrase
        batch_label = batch.Sentiment
        out = model(batch_text)  # [batch_size, label_num]
        loss = criterion(out, batch_label)
        total_loss = total_loss + loss.item()

        loss.backward()
        optimizer.step()

        correct = (torch.max(out, dim=1)[1]  # get the indices
                   .view(batch_label.size()) == batch_label).sum()
        total_correct = total_correct + correct.item()

        print("Epoch", epoch, end=':')
        print(' ', int(steps * BATCH_SIZE), '/', total_data_num, end='')
        print(' Training average Loss: ', total_loss / steps)

            # 验证
    model.eval()
    total_loss = 0.0
    accuracy = 0.0
    total_correct = 0.0
    total_data_num = len(dev_iterator.dataset)
    steps = 0.0
    for batch in dev_iterator:
        steps += 1
        batch_text = batch.Phrase
        batch_label = batch.Sentiment
        out = model(batch_text)
        loss = criterion(out, batch_label)
        total_loss = total_loss + loss.item()

        correct = (torch.max(out, dim=1)[1].view(batch_label.size()) == batch_label).sum()
        total_correct = total_correct + correct.item()

        print("Epoch %d :  Verification average Loss: %f, Verification accuracy: %f%%,Total Time:%f"
              % (epoch, total_loss / steps, total_correct * 100 / total_data_num, time.time() - start))

        if best_accuracy < total_correct / total_data_num:
            best_accuracy = total_correct / total_data_num
            torch.save(model, 'model_dict/model_lstm/epoch_%d_accuracy_%f' % (epoch, total_correct / total_data_num))
            print('Model is saved in model_dict/model_lstm/epoch_%d_accuracy_%f' % (epoch, total_correct / total_data_num))
            # torch.cuda.empty_cache()
    # break  # 训练时去掉

# # 测试-重新读取文件（方便重写成.py文件）
# PATH = 'model_dict/model_lstm/epoch_5_a'
# model = torch.load(PATH)
#
# total_loss = 0.0
# accuracy = 0.0
# total_correct = 0.0
# total_data_num = len(train_iterator.dataset)
# steps = 0.0
# start_time = time.time()
# for batch in test_iterator:
#     steps += 1
#     batch_text = batch.Phrase
#     batch_label = batch.Sentiment
#     out = model(batch_text)
#     loss = criterion(out, batch_label)
#     total_loss = total_loss + loss.item()
#
#     correct = (torch.max(out, dim=1)[1].view(batch_label.size()) == batch_label).sum()
#     total_correct = total_correct + correct.item()
#
# print("Test average Loss: %f, Test accuracy: %f，Total time: %f"
#       % (total_loss / steps, total_correct / total_data_num, time.time() - start_time))
