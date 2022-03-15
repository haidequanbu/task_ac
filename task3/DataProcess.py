import pickle
import numpy as np
import string
import argparse
import json


class DataProcess(object):

    def read_data(self, data_dir):
        premise = []
        hypothesis = []
        labels = []
        labels_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
        punct_table = str.maketrans({key: " " for key in string.punctuation})
        with open(data_dir, 'r', encoding='utf-8') as lines:
            next(lines)
            for line in lines:
                line = line.strip().split('\t')
                if line[0] not in labels_map:  # 忽略没有label的例子
                    continue
                premise.append(line[5].translate(punct_table).lower())
                hypothesis.append(line[6].translate(punct_table).lower())
                labels.append(line[0])
        return {"premise": premise,
                "hypothesis": hypothesis,
                "labels": labels}

    def build_worddict(self, data, worddict_dir):
        words = []
        words.extend(["_PAD_", "_OOV_", "_BOS_", "_EOS_"])
        for sentence in data["premise"]:
            words.extend(sentence.strip().split(" "))
        for sentence in data["hypothesis"]:
            words.extend(sentence.strip().split(" "))
        word_id = {}
        id_word = {}
        i = 0
        for index, word in enumerate(words):
            if word not in word_id:
                word_id[word] = i
                id_word[i] = word
                i += 1
        # 保存词典
        with open(worddict_dir, "w", encoding='utf-8') as f:
            for word in word_id:
                f.write("%s\t%d\n" % (word, word_id[word]))
        return word_id, id_word

    def sentence2idList(self, sentence, word_id):
        ids = []
        ids.append(word_id["_BOS_"])
        sentence = sentence.strip().split(" ")
        for word in sentence:
            if word not in word_id:
                ids.append(word_id["_OOV_"])
            else:
                ids.append(word_id[word])
        ids.append(word_id["_EOS_"])
        return ids

    def data2id(self, data, word_id):
        premise_id = []
        hypothesis_id = []
        labels_id = []
        labels_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
        for i, label in enumerate(data["labels"]):
            if label not in labels_map:  # 忽略没有label的例子
                continue
            premise_id.append(self.sentence2idList(data["premise"][i], word_id))
            hypothesis_id.append(self.sentence2idList(data["hypothesis"][i], word_id))
            labels_id.append(labels_map[label])

        return {"premises": premise_id,
                "hypothesis": hypothesis_id,
                "labels": labels_id}

    def build_embeddings(self, embedding_file, word_id):
        # 读取文件存入集合中
        embeddings_map = {}
        with open(embedding_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split()
                word = line[0]
                if word in word_id:
                    embeddings_map[word] = line[1:]
                    # 放入矩阵中
        words_num = len(word_id)
        embedding_dim = len(embeddings_map['a'])
        embedding_matrix = np.zeros((words_num, embedding_dim))
        # print(words_num,embedding_dim)
        missed_cnt = 0
        for i, word in enumerate(word_id):
            if word in embeddings_map:
                embedding_matrix[i] = embeddings_map[word]
            else:
                if word == "_PAD_":
                    continue
                missed_cnt += 1
                embedding_matrix[i] = np.random.normal(size=embedding_dim)
        print("missed word count: %d" % (missed_cnt))
        return embedding_matrix


if __name__ == '__main__':
    # 配置文件地址
    default_config = 'config/preProcess_config.json'

    # 命令行参数
    parser = argparse.ArgumentParser(description='config of dataset preprocess!')
    parser.add_argument("--config", default=default_config)
    args = parser.parse_args()

    # 读取配置文件
    with open(args.config, 'r') as f:
        config = json.load(f)

    # 配置文件中读取配置
    data_train_dir = config["data_train_dir"]
    data_dev_dir = config["data_dev_dir"]
    embedding_file_dir = config["embedding_file_dir"]

    worddict_dir = config["worddict_dir"]
    data_train_str_dir = config["data_train_str_dir"]
    data_train_id_dir = config["data_train_id_dir"]
    data_dev_str_dir = config["data_dev_str_dir"]
    data_dev_id_dir = config["data_dev_id_dir"]
    embedding_matrix_dir = config["embedding_matrix_dir"]

    print("-" * 30, "starting preprocessing data!", "-" * 30)

    data_processor = DataProcess()

    # 读取数据
    data_str = data_processor.read_data(data_dir=data_train_dir)
    data_dev_str = data_processor.read_data(data_dir=data_dev_dir)
    # 构建词典
    word_id, id_word = data_processor.build_worddict(data_str, worddict_dir)
    # 清洗数据并转换为id
    data_id = data_processor.data2id(data_str, word_id)
    data_id_dev = data_processor.data2id(data_dev_str, word_id)

    # 保存 data_train_str和data_train_id
    with open(data_train_str_dir, "wb") as f:
        pickle.dump(data_str, f)
    with open(data_train_id_dir, "wb") as f:
        pickle.dump(data_id, f)
    with open(data_dev_str_dir, "wb") as f:
        pickle.dump(data_dev_str, f)
    with open(data_dev_id_dir, "wb") as f:
        pickle.dump(data_id_dev, f)

    # embedding矩阵的生成和保存
    embedding_matrix = data_processor.build_embeddings(embedding_file_dir, word_id)
    print("embedding_matrix size: %d" % len(embedding_matrix))
    with open(embedding_matrix_dir, "wb") as f:
        pickle.dump(embedding_matrix, f)

    print("-" * 30, "preprocessing data finished!", "-" * 30)
