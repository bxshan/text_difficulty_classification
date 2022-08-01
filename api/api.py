import json
import os
import pickle
import random
import re
import subprocess
import sys
from os.path import dirname, isabs, join

import nltk
import numpy as np
from nltk import pos_tag, regexp_tokenize, word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from six import BytesIO
from sklearn.linear_model import LogisticRegression

path_data = join(dirname(os.path.abspath(__file__)), 'nltk_data')
nltk.data.path.append(path_data)

localpool = dirname(dirname(__file__))
CLASS_NUM, WORD_DIFF_NUM = 4, 4  # 全局变量，代表难度分级和单词难度分级，一般来说都是一个数。


def readtext(filepath):  # 读取文本
    if not isabs(filepath):
        filepath = join(localpool, filepath)
    with open(filepath, 'r', encoding="utf-8") as f:
        text = f.read()
    return text


def writetext(text, filepath):  # 修改某个.txt
    if not isabs(filepath):
        filepath = join(localpool, filepath)
    with open(filepath, 'w', encoding="utf8") as f:
        f.write(text)


def get_wordnet_pos(treebank_tag):  # 和wordNET的词性有关
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def splitwords(sentence):  # 将句子分割成单词并且获得词根
    res = []
    lemmatizer = WordNetLemmatizer()  # 分词器
    for word, pos in pos_tag(regexp_tokenize(sentence.lower(), r'\w+')):
        wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN
        res.append(lemmatizer.lemmatize(word, pos=wordnet_pos))  # 去掉变形，返回词根
    return res


def load_data(listpath, label_choose={}):  # 加载数据，是下面三个加载各种数据的基函数
    with open(listpath) as f:
        lines = f.readlines()

    data = []
    for line in lines:
        path, label = line.strip().split()
        label = int(label)
        if label in label_choose:
            data.append([path, label])
    return data


def load_textbooks_data():
    data = load_data(join(localpool, 'data/textbooks/textlist.txt'),
                     set(range(CLASS_NUM)))
    return data


def load_train_data():
    data = load_data(join(localpool, 'data/reading/trainlist.txt'),
                     set(range(CLASS_NUM)))
    return data


def load_test_data():
    data = load_data(join(localpool, 'data/reading/testlist.txt'),
                     set(range(CLASS_NUM)))
    return data


def mean_3_sigma(li):  # 类似于正态分布，去掉3\sigma之外的样本。
    _mean = np.mean(li)
    _std = np.std(li)
    tmp = [i for i in li if _mean - 3 * _std <= i <= _mean + 3 * _std]

    return tmp


# Get the word diff level
def get_diff_level(path_grade):
    diff_level = {}
    for path, grade in path_grade:
        text = readtext(path)
        words = splitwords(text)
        grade = int(grade)
        for word in words:
            if word in diff_level and diff_level[word] <= grade:
                continue
            else:
                diff_level[word] = grade
    return diff_level


def features_word_diff(path, diff_level, method=1):  # 获取词汇难度的方法，是最后返回词汇难度的频率。
    # 如果method=1，则使用前面的根据新概念课文中的出现最高难度来定义单词难度。
    # 如果method不为1，则使用NGSL频率表。

    text = readtext(path)
    words = splitwords(text)
    if method == 1:  # 老方法
        grade_freq = [0] * WORD_DIFF_NUM
        for word in words:
            if word in diff_level:
                grade = diff_level[word]
                grade_freq[grade] += 1
            else:
                continue
        num = sum(grade_freq)
        for i in range(WORD_DIFF_NUM):
            grade_freq[i] /= num
        return grade_freq

    else:
        total = 39099970
        nums = 22211  # 这两个参数是NGSL中去掉前二十个高频词之后的总结果，total是出现总频率，nums是单词种类数。
        dict_path = 'api/diff_dict.pkl'  # 存放路径。注意这里是相对于主函数的路径。
        f = open(dict_path, 'rb')
        diff_dict = pickle.load(f)
        f.close()

        grade_freq = 0
        for word in words:
            if diff_dict.__contains__(word):
                grade_freq += diff_dict[word]
            else:
                continue  # 当然也有漏掉的词和那些被主动去掉的高频词，所以需要continue。

        return [grade_freq * nums / total / 10000]  # 返回平均值，系数可以调。


def get_feats_labels(data,
                     diff_level,
                     newFeatures=None,
                     diff_use=1,
                     add_len=True):  # 获得特征
    # diff_use就是两种难度评级
    features = []
    labels = []
    for path, label in data:
        features.append(features_word_diff(path, diff_level,
                                           diff_use))  # 这里首先是单拎出来的词汇难度特征。

        if not (newFeatures is None):
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            for F in newFeatures:  # 遍历特征列表里每一种需要提取的特征。
                tmp_f = F(text)
                # 下面的有关对齐的讨论比较繁琐，可以简化，就可以直接理解为每次把新提取的特征缝到一起
                if not add_len:
                    if len(tmp_f) > len(features[-1]):
                        features[-1] += [0] * (len(tmp_f) - len(features[-1]))
                    features[-1] = [
                        features[-1][i] + tmp_f[i]
                        for i in range(len(features[-1]))
                    ]
                else:
                    features[-1] += tmp_f

        features[-1] = [i / sum(features[-1])
                        for i in features[-1]]  # 输出features[-1]即可看到特征。
        # [-1]代表队尾，feature这个list包含所有数据集的feature，120个或者40个，[-1]代表最新添加的一个。

        labels.append(int(label))  # label是'1'这种字符串形式的数字，注意转化。

    return features, labels


def shuffle_data(n=10):  # 重新分配训练集和测试集
    prefix = 'data/reading/dataset/'
    diff_list = [i for i in range(0, 40)]

    test_f = open('data/reading/testlist.txt', 'w')
    train_f = open('data/reading/trainlist.txt', 'w')
    for diff in range(0, 4):
        test_list = random.sample(diff_list, n)
        train_list = list(set(diff_list) - set(test_list))
        for num in test_list:
            test_f.write(prefix + str(diff) + '-' + str(num) + '.txt ' +
                         str(diff) + "\n")
        for num in train_list:
            train_f.write(prefix + str(diff) + '-' + str(num) + '.txt ' +
                          str(diff) + "\n")

    test_f.close()
    train_f.close()


class logistic_regression():  # 逻辑回归原型。
    def __init__(self, optimizer='logistic'):
        if optimizer == 'logistic':
            self.cls = LogisticRegression(C=10000,
                                          max_iter=10000,
                                          multi_class='ovr')  # 初始化逻辑回归。
        else:
            self.cls = None

        self.x = None
        self.y = None
        self.weights = []
        self.optimizer = optimizer

    def make_data_x(self, x):
        x = np.array(x)
        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=1)

        return x

    def make_data_y(self, y):
        y = np.array(y)

        return y

    def train(self, x, y):
        self.x = self.make_data_x(x)
        self.y = self.make_data_y(y)
        self.cls.fit(self.x, self.y)  #进行拟合

    def pred(self, x):  # 做预测的函数。
        if type(x) is not np.ndarray:
            x = self.make_data_x(x)
        ret = self.cls.predict(x).tolist()

        if len(ret) == 1:
            ret = ret[0]

        return ret


class order_regression():  # 和上面的大同小异
    def __init__(self):
        self.label_num = 0
        self.classifier_num = 0

    def train(self, x, y):
        x = np.array(x)
        y = np.array(y)
        label_name = sorted(list(set(y)))
        self.label_num = len(label_name)
        self.classifier_num = self.label_num - 1
        self.classifiers = list()
        for i in range(self.classifier_num):
            c_i = LogisticRegression(C=10000, max_iter=10000)
            l_0 = np.where(y < label_name[i + 1])
            label_tmp = np.ones(len(y))
            label_tmp[l_0] = 0
            # ---------------------
            # [1,2,3,5,4,2,2] ->
            # [0,0,1,1,1,0,0]
            # ---------------------
            c_i.fit(x, label_tmp)
            self.classifiers.append(c_i)

    def pred(self, x):
        x = np.array(x)
        out = np.zeros(x.shape[0])
        for c_i in self.classifiers:
            tmp_y = c_i.predict_proba(x)
            out += tmp_y[:, 1]
        return np.round(out)


def accuracy(pred, y):  # 评估准确率的函数。
    right = 0
    total = 0
    for i in range(len(pred)):
        if pred[i] == y[i]:
            right += 1
        total += 1

    acc = 1.0 * right / total
    return acc
