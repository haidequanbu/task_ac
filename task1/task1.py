import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from sklearn.naive_bayes import MultinomialNB

def extract_data():
    # 准备数据，划分测试集，训练集
    ratings=pd.read_table('data/train.tsv')
    x_all=ratings['Phrase']
    y_all=ratings['Sentiment']
    x_train,x_test,y_trian,y_test=train_test_split(x_all,y_all,test_size=0.2)
    x_train,x_val,y_trian,y_val=train_test_split(x_train,y_trian,test_size=0.25)
    count_vect=CountVectorizer()
    x_train_counts=count_vect.fit_transform(x_train)
    x_test_counts=count_vect.transform(x_test)
    print("训练集和测试集大小(集合大小，特征维度）：", end='')
    print(x_train_counts.shape,x_test_counts.shape)

    # 提取一维单词Tf-idf特称
    tfidf_transformer=TfidfVectorizer(analyzer='word',max_features=20000)
    tfidf_transformer.fit(x_train)
    x_train_tfidf_word=tfidf_transformer.transform(x_train)
    x_test_tfidf_word=tfidf_transformer.transform(x_test)
    print("单个单词的大小（集合大小，特征维度）：",end='')
    print(x_train_tfidf_word.shape, x_test_tfidf_word.shape)

    # 提取n-gram级别的Tf-idf特征
    tfidf_transformer2=TfidfVectorizer(analyzer='word',ngram_range=(2,3),max_features=50000)
    tfidf_transformer2.fit(x_train)
    x_train_tfidf_ngram=tfidf_transformer2.transform(x_train)
    x_test_tfidf_ngram=tfidf_transformer2.transform(x_test)
    print("2-3个Ngram的最大特征分量为50000是大小（集合大小，特征维度）：", end='')
    print(x_train_tfidf_ngram.shape,x_test_tfidf_ngram.shape)

    # 合并特征，获取训练基础数据
    # train_features=x_train_counts
    # test_features=x_test_counts
    train_features=hstack([x_train_counts,x_train_tfidf_word,x_train_tfidf_ngram])
    test_features=hstack([x_test_counts,x_test_tfidf_word,x_test_tfidf_ngram])
    print("训练模型数据：（集合大小，特征维度）：",end='')
    print(train_features.shape,test_features.shape)
    return train_features,test_features,y_trian,y_test


if __name__=='__main__':
    # 准备数据
    train_features,test_features ,y_train,y_test= extract_data()
    train_features, y_train = shuffle(train_features, y_train)

    # 逻辑回归,准确率
    log_model = LogisticRegression(random_state=0, solver='sag', multi_class='multinomial',max_iter=120)
    log_model.fit(train_features, y_train)
    predict_log=log_model.predict(test_features)
    print(np.mean(predict_log==y_test))

    # 朴素贝叶斯，准确率
    nb_model=MultinomialNB()
    nb_model.fit(train_features, y_train)
    predict_nb = nb_model.predict(test_features)
    print(np.mean(predict_nb == y_test))

    # SVM，准确率
    svm_model=SGDClassifier(alpha=0.001,loss='log',early_stopping=True,eta0=0.001,learning_rate='adaptive',max_iter=100)
    svm_model.fit(train_features, y_train)
    predict_svm = svm_model.predict(test_features)
    print(np.mean(predict_svm == y_test))