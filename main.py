import torch
import torch.nn as nn
from Instance import Inst
from example import Example
from AlphaBet import AlphaBet
from hyperparameter import Hyperparameter
import re
import numpy as np
import random
import sys
import os
import collections
import pickle

class Classifier:
    def __init__(self):
        self.word_state = collections.OrderedDict()
        self.word_AlphaBet = AlphaBet()
        self.label_AlphaBet = AlphaBet()
        self.hyperpara = Hyperparameter()
    def read_file(self, path):
        f = open(path)
        L = []
        # num = 0
        for line in f.readlines():
            result = Inst()
            # num += 1
            info = line.strip().split('|||')
            info[0] = self.clean_str(info[0])
            result.m_word = info[0].strip().split(' ')
            result.m_label = info[1].strip()
            L.append(result)
            # num += 1
            # if num == 800:
            #     break
        f.close()
        return L
    def clean_str(self, string):

        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip()

    def extract_feature(self, result):
        all_word = []
        all_label = []
        all_word_feature = []
        all_label_indexes = []
        for i in result:
            m_word_feature = []
            for idx in range(len(i.m_word)):
                a = 'unigram = ' + i.m_word[idx]
                # inst.m_word.append(a)
                m_word_feature.append(a)
                all_word_feature.append(a)
            for idx in range(len(i.m_word) - 1):
                a = 'bigram = ' + i.m_word[idx] + '#' + i.m_word[idx + 1]
                # inst.m_word.append(a)
                m_word_feature.append(a)
                all_word_feature.append(a)
            for idx in range(len(i.m_word) - 2):
                a = 'trigram = ' + i.m_word[idx] + '#' + i.m_word[idx + 1] + '#' + i.m_word[idx + 2]
                m_word_feature.append(a)
                all_word_feature.append(a)   #所有的特征
                # inst.m_word.append(a)
            all_word.append(m_word_feature)   #以每句为单位的特征
            all_label.append(i.m_label)

            if i.m_label not in all_label_indexes:
                all_label_indexes.append(i.m_label)

        return all_word, all_label, all_word_feature, all_label_indexes

    def create_AlphaBet(self, all_word_feature):
        print("create dict-------------------------")
        word_state = []
        for m in all_word_feature:
            if m not in self.word_state:
                word_state.append(m)
        # word_state.append(self.hyperpara.unknow)
        self.word_AlphaBet.makeVocab(word_state)
        # self.hyperpara.unknow_id = self.word_AlphaBet.dict[self.hyperpara.unknow]
        return self.word_AlphaBet

    def change(self, file_train):
        i, j, x, y = self.extract_feature(file_train)
        all_examples = []
        for idx in range(len(i)):
            m = i[idx]
            example = Example()
            for a in m:
                if a in self.word_AlphaBet.dict:
                    example.m_word_indexes.append(self.word_AlphaBet.dict[a])
            label_list = [0, 0, 0, 0, 0]
            b = int(j[idx])
            label_list[b] = 1
            example.m_label_index = label_list
            all_examples.append(example)
        return all_examples

    def Init_weight_array(self, all_word_feature):
        w_feature = self.create_AlphaBet(all_word_feature)
        self.weight_array = np.random.rand(len(w_feature.list), self.hyperpara.class_num)
        return self.weight_array

    def getMaxIndex(self, result):
        max = np.max(result)
        for idx in range(len(result)):
            if result[idx] == max:
                return idx

    def Y_list(self, all_examples):
        y_list = []
        for i in all_examples:
            m_result = [0.0, 0.0, 0.0, 0.0, 0.0]
            for j in i.m_word_indexes:
                m_result += np.array(self.weight_array[j])
            y_list.append(m_result)
        return y_list

    def set_batchBlock(self, examples):
        if len(examples) % self.hyperpara.batch_size == 0:
            batchBlock = len(examples) // self.hyperpara.batch_size
        else:
            batchBlock = len(examples) // self.hyperpara.batch_size + 1
        return batchBlock

    def count_loss(self, y):
        p = np.max(y)
        loss = -1 * np.log(p)
        return loss

    def softmax(self, result):
        result_list = []
        bottom = 0
        max_idx = self.getMaxIndex(result)
        for index, value in enumerate(result):
            bottom += np.exp(value - result[max_idx])
        for index, value in enumerate(result):
            result_list.append(np.exp(value - result[max_idx])/bottom)
        return result_list

    def train(self, path_train, path_dev, path_test):
        print("train start ......")

        file_train = self.read_file(path_train)   #train_inst
        file_dev = self.read_file(path_dev)
        file_test = self.read_file(path_test)
        m_train, l_index, w_train, l_train = self.extract_feature(file_train)
        # w_alpha, l_bet = self.create_AlphaBet(w_train, l_train)
        w_alpha = self.create_AlphaBet(w_train)         #feature_alphabet

        e_train = self.change(file_train)              #train_exam_list
        e_dev = self.change(file_dev)                  #m_word_indexes, m_label_index
        e_test = self.change(file_test)
        # self.weight_array = self.Init_weight_array(w_train, l_train)
        self.weight_array = self.Init_weight_array(w_train)
        for epoch in range(1, self.hyperpara.epochs + 1):
            print("————————第{}轮迭代，共{}轮————————".format(epoch, self.hyperpara.epochs))
            batchBlock = self.set_batchBlock(e_train)
            random.shuffle(e_train)
            m_result = self.Y_list(e_train)  # y = w*x
            train_size = len(m_result)
            corrects, acc, sum, steps, all_loss = 0, 0, 0, 0, 0
            for every_batchBlock in range(batchBlock):
                exam = []
                start_pos = every_batchBlock * self.hyperpara.batch_size
                end_pos = (every_batchBlock + 1) * self.hyperpara.batch_size
                if end_pos > len(e_train):
                    end_pos = len(e_train)
                init_grad_w = np.zeros((len(w_alpha.list), self.hyperpara.class_num))
                for idx in range(start_pos, end_pos):
                    steps += 1
                    sum += 1
                    y = self.softmax(m_result[idx])
                    each_loss = self.count_loss(y)
                    all_loss += each_loss
                    pd_l_for_y = np.subtract(y, e_train[idx].m_label_index)
                    pd_l_for_y = pd_l_for_y / self.hyperpara.batch_size
                    predict_label = self.getMaxIndex(m_result[idx])
                    real_label = self.getMaxIndex(e_train[idx].m_label_index)
                    if predict_label == real_label:
                        corrects += 1
                    for z in e_train[idx].m_word_indexes:
                        init_grad_w[z] += pd_l_for_y
                        if z not in exam:
                            exam.append(z)
                    for i in exam:
                        self.weight_array[i] -= self.hyperpara.lr * np.array(init_grad_w[i])
                acc = corrects / sum * 100.0
            if steps % self.hyperpara.log_interval == 0:
                sys.stdout.write(
                    '\rBatch[{}/{}]- loss:{:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                            train_size,
                                                                            all_loss,
                                                                            acc,
                                                                            corrects,
                                                                            sum))
            if steps % self.hyperpara.test_interval == 0:
                self.eval(e_dev)

            # if steps % self.hyperpara.save_interval == 0:
            #     if not os.path.isdir(self.hyperpara.save_dir):os.makedirs(self.hyperpara.save_dir)
            #     save_prefix = os.path.join(self.hyperpara.save_dir, 'snapshot')
            #     save_path = '{}_steps{}.pt'.format(save_prefix, steps)
            #     np.save(save_path, ifier)
        result_list = []
        if os.path.exists("./Test_Result.txt"):
            file = open("./Test_Result.txt")
            for line in file.readlines():
                if line[:10] == "Evaluation":
                    result_list.append(float(line[19:25]))
            result = sorted(result_list)
            file.close()
            file = open("./Test_Result.txt", "a")
            file.write("The best result is :" + str(result[len(result) - 1]))
            file.write("\n \n")
            file.close()

        return e_train, e_dev, e_test

    def eval(self, e_dev):
        corrects, acc, sum = 0, 0, 0
        m_result = self.Y_list(e_dev)
        dev_size = len(m_result)
        for idx in range(dev_size):
            y = self.softmax(m_result[idx])
            num = self.getMaxIndex(y)
            label_num = self.getMaxIndex(e_dev[idx].m_label_index)
            if num == label_num:
                corrects += 1
            sum += 1
        acc = corrects / sum * 100.0

        print('\nEvaluation - acc: {:.4f}%({}/{})]\n'.format(acc,
                                                             corrects,
                                                             dev_size))
        if os.path.exists("./Test_Result.txt"):
            file = open("./Test_Result.txt", "a")
        else:
            file = open("./Test_Result.txt", "w")
        file.write('\nEvaluation -  acc: {:.4f}%({}/{}) \n'.format(acc,
                                                                   corrects,
                                                                   dev_size))
        file.close()
path_dev = './data/raw.clean.dev'
path_train = './data/raw.clean.train'
path_test = './data/raw.clean.test'
# path_train = './data/simple.train'
# path_dev = './data/simple.train'
# path_test = './data/simple.train'
ifier = Classifier()
ifier.train(path_train, path_dev, path_test)

