import os
import re
import random
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from math import log
from collections import defaultdict
import jieba
from tqdm import tqdm
import sys
from pathlib import Path
from transformers import BertTokenizer

global_words_freq = defaultdict(int)
min_words_freq = 15
word_embeddings_dim = 768

tokenizer = BertTokenizer.from_pretrained('./pretrained_models/Roberta-mid')

if len(sys.argv) != 2:
	sys.exit("Use: python build_graph.py <dataset>")

datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr', 'iflytek', 'thucnews', 'inews', 'fudan', 'sogoucs']
dataset = sys.argv[1]

if dataset not in datasets:
	sys.exit("wrong dataset name")

doc_name_list = []
doc_train_list = []
doc_test_list = []
f_doc_list = open('./data/text_dataset/' + dataset + '.txt', 'r')
lines = f_doc_list.readlines()
for line in tqdm(lines):
    doc_name_list.append(line.strip())
    temp = line.split("\t")
    if temp[1].find('test') != -1:
        doc_test_list.append(line.strip())
    elif temp[1].find('train') != -1:
        doc_train_list.append(line.strip())
f_doc_list.close()

doc_content_list = []
f_content = open('./data/clean_corpus/' + dataset + '.clean.txt', 'r', encoding='utf-8')
lines = f_content.readlines()
for line in tqdm(lines):
    words = tokenizer.tokenize(line)
    get_result = []
    for word in words:
        global_words_freq[word] += 1
        if re.match("[\u4e00-\u9fa5]+", word):
            get_result.append(word)
    doc_content_list.append(get_result)
f_content.close()

train_ids = []
for train_name in tqdm(doc_train_list):
    train_id = doc_name_list.index(train_name)
    train_ids.append(train_id)
random.shuffle(train_ids)

train_ids_str = '\n'.join(str(index) for index in train_ids)
f_train = open('./data/graph/' + dataset + '.train.index', 'w')
f_train.write(train_ids_str)
f_train.close()

test_ids = []
for test_name in tqdm(doc_test_list):
    test_id = doc_name_list.index(test_name)
    test_ids.append(test_id)
random.shuffle(test_ids)

test_ids_str = '\n'.join(str(index) for index in test_ids)
f_test = open('./data/graph/' + dataset + '.test.index', 'w')
f_test.write(test_ids_str)
f_test.close()

ids = train_ids + test_ids

shuffle_doc_name_list = []
shuffle_doc_words_list = []
for id in tqdm(ids):
    shuffle_doc_name_list.append(doc_name_list[int(id)])
    shuffle_doc_words_list.append(doc_content_list[int(id)])

shuffle_doc_name_str = '\n'.join(shuffle_doc_name_list)

f_shuffle_doc_name_str = open('./data/graph_data/' + dataset + '_shuffle.txt', 'w')
f_shuffle_doc_name_str.write(shuffle_doc_name_str)
f_shuffle_doc_name_str.close()

f_shuffle_doc_words_str = open('./data/graph_data/clean_corpus/' + dataset + '_shuffle.txt', 'w')
for sublist in tqdm(shuffle_doc_words_list):
    shuffle_doc_words_str = ''.join(sublist).rstrip()
    f_shuffle_doc_words_str.write(shuffle_doc_words_str + '\n')
f_shuffle_doc_words_str.close()

print('build vocab')
word_set = set()

with open('./data/stopwords/new_chinese_stopwords.txt', 'r', encoding='utf-8') as f_stop:
    stopword_list = [line.strip() for line in f_stop.readlines()]
f_stop.close()

for doc_words in tqdm(shuffle_doc_words_list):
    for word in doc_words:
        if word not in stopword_list and global_words_freq[word] >= min_words_freq:
        word_set.add(word)

vocab = list(word_set)
vocab_size = len(vocab)
print('vocab_size:{}'.format(vocab_size))

print('build word_doc_list.')
word_doc_list = defaultdict(int)
for i in tqdm(range(len(shuffle_doc_words_list))):
    doc_words = shuffle_doc_words_list[i]
    appeared = set()
    for word in doc_words:
        if word in appeared:
            continue
        if word in word_doc_list:
            doc_list = word_doc_list[word]
            doc_list.append(i)
            word_doc_list[word] = doc_list
        else:
            word_doc_list[word] = [i]
        appeared.add(word)
print('word_doc_list_length:{}'.format(len(word_doc_list)))

print('count word in document frequency')
word_doc_freq = defaultdict(int)
for word, doc_list in tqdm(word_doc_list.items()):
    word_doc_freq[word] = len(doc_list)
print('word_doc_freq_length:{}'.format(len(word_doc_freq)))

print('build word_id_map.')
word_id_map = defaultdict(int)
for i in tqdm(range(vocab_size)):
    word_id_map[vocab[i]] = i
print('word_id_map_length:{}'.format(len(word_id_map)))

vocab_str = '\n'.join(vocab)
f_vocab_str = open('./data/graph_data/clean_corpus/' + dataset + '_vocab.txt', 'w')
f_vocab_str.write(vocab_str)
f_vocab_str.close()

label_set = set()
for doc_meta in shuffle_doc_name_list:
    temp = doc_meta.split('\t')
    label_set.add(temp[2])
label_list = list(label_set)
print('label_list_length:{}'.format(len(label_list)))

label_list_str = '\n'.join(label_list)
f_label_list_str = open('./data/graph_data/clean_corpus/' + dataset + '_labels.txt', 'w')
f_label_list_str.write(label_list_str)
f_label_list_str.close()

train_size = len(train_ids)
val_size = int(0.1 * train_size)
real_train_size = train_size - val_size
test_size = len(test_ids)

# different training rates
print('data_size:{}'.format(str(train_size + test_size)))
print('train_size:{}'.format(str(train_size)))
print('val_size:{}'.format(str(val_size)))
print('real_train_size:{}'.format(str(real_train_size)))
print('test_size:{}'.format(str(test_size)))

real_train_doc_names = shuffle_doc_name_list[:real_train_size]
real_train_doc_names_str = '\n'.join(real_train_doc_names)

f = open('./data/graph_data/' + dataset + '.real_train.name', 'w')
f.write(real_train_doc_names_str)
f.close()

print('build x.')
row_x = []
col_x = []
data_x = []

for i in tqdm(range(real_train_size)):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i]
    doc_len = len(doc_words)
    if doc_len > 0:
        for j in range(word_embeddings_dim):
            row_x.append(int(i))
            col_x.append(j)
            data_x.append((doc_vec[j] / doc_len) if not np.isinf(doc_vec[j] / doc_len) and not np.isnan(doc_vec[j] / doc_len) else 0)
    else:
        for j in range(word_embeddings_dim):
            row_x.append(int(i))
            col_x.append(j)
            data_x.append(0)

x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
    real_train_size, word_embeddings_dim))

print('build y.')
y = []
for i in tqdm(range(real_train_size)):
    doc_meta = shuffle_doc_name_list[i]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    y.append(one_hot)
y = np.array(y)

print('build tx.')
row_tx = []
col_tx = []
data_tx = []
for i in tqdm(range(test_size)):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i + train_size]
    doc_len = len(doc_words)
    if doc_len > 0:
        for j in range(word_embeddings_dim):
            row_tx.append(int(i))
            col_tx.append(j)
            data_tx.append((doc_vec[j] / doc_len) if not np.isinf(doc_vec[j] / doc_len) and not np.isnan(doc_vec[j] / doc_len) else 0)
    else:
        for j in range(word_embeddings_dim):
            row_tx.append(int(i))
            col_tx.append(j)
            data_tx.append(0)

tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
                   shape=(test_size, word_embeddings_dim))

print('build ty.')
ty = []
for i in tqdm(range(test_size)):
    doc_meta = shuffle_doc_name_list[i + train_size]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ty.append(one_hot)
ty = np.array(ty)

word_vectors = np.random.uniform(-0.01, 0.01, (vocab_size, word_embeddings_dim))

print('build allx.')
row_allx = []
col_allx = []
data_allx = []

for i in tqdm(range(train_size)):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i]
    doc_len = len(doc_words)
    if doc_len > 0:
        for j in range(word_embeddings_dim):
            row_allx.append(int(i))
            col_allx.append(j)
            data_allx.append((doc_vec[j] / doc_len) if not np.isinf(doc_vec[j] / doc_len) and not np.isnan(doc_vec[j] / doc_len) else 0)
    else:
        for j in range(word_embeddings_dim):
            row_allx.append(int(i))
            col_allx.append(j)
            data_allx.append(0)

for i in tqdm(range(vocab_size)):
    for j in range(word_embeddings_dim):
        row_allx.append(int(i + train_size))
        col_allx.append(j)
        data_allx.append(word_vectors.item((i, j)))

row_allx = np.array(row_allx)
col_allx = np.array(col_allx)
data_allx = np.array(data_allx)

allx = sp.csr_matrix(
    (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, word_embeddings_dim))

print('build ally.')
ally = []
for i in tqdm(range(train_size)):
    doc_meta = shuffle_doc_name_list[i]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ally.append(one_hot)

for i in tqdm(range(vocab_size)):
    one_hot = [0 for l in range(len(label_list))]
    ally.append(one_hot)

ally = np.array(ally)

print('x.shape:{}, y.shape:{}, tx.shape:{}, ty.shape:{}, allx.shape:{}, ally.shape:{}'.format(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape))

window_size = 20
windows = []

print('build window')
for doc_words in tqdm(shuffle_doc_words_list):
    length = len(doc_words)
    if length <= window_size:
        windows.append(doc_words)
    else:
        for j in range(length - window_size + 1):
            window = doc_words[j: j + window_size]
            windows.append(window)
print('window_length:{}'.format(len(windows)))

print('count word in window frequency')
word_window_freq = defaultdict(int)
for window in tqdm(windows):
    appeared = set()
    for i in range(len(window)):
        if window[i] in appeared:
            continue
        if window[i] in word_window_freq:
            word_window_freq[window[i]] += 1
        else:
            word_window_freq[window[i]] = 1
        appeared.add(window[i])
print('word_window_freq_length:{}'.format(len(word_window_freq)))

print('count vocabulary and vocabulary frequency')
word_pair_count = defaultdict(int)
for window in tqdm(windows):
    for i in range(1, len(window)):
        for j in range(0, i):
            word_i = window[i]
            word_i_id = word_id_map[word_i]
            word_j = window[j]
            word_j_id = word_id_map[word_j]
            if word_i_id == word_j_id:
                continue
            word_pair_str = str(word_i_id) + ',' + str(word_j_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1
            # two orders
            word_pair_str = str(word_j_id) + ',' + str(word_i_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1
print('word_pair_count_length:{}'.format(len(word_pair_count)))

print('build adjacency_matrix in vocabulary')
rows = []
cols = []
weight = []

num_window = len(windows)

for key in tqdm(word_pair_count):
    temp = key.split(',')
    i = int(temp[0])
    j = int(temp[1])
    count = word_pair_count[key]
    word_freq_i = word_window_freq[vocab[i]]
    word_freq_j = word_window_freq[vocab[j]]
    pmi = log((1.0 * count / num_window) /
              (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
    if pmi <= 0:
        continue
    rows.append(train_size + i)
    cols.append(train_size + j)
    weight.append(pmi)

# doc word frequency
print('count document in word frequency')
doc_word_freq = defaultdict(int)
for doc_id in tqdm(range(len(shuffle_doc_words_list))):
    doc_words = shuffle_doc_words_list[doc_id]
    for word in doc_words:
        word_id = word_id_map[word]
        doc_word_str = str(doc_id) + ',' + str(word_id)
        if doc_word_str in doc_word_freq:
            doc_word_freq[doc_word_str] += 1
        else:
            doc_word_freq[doc_word_str] = 1
print('doc_word_freq_length:{}'.format(len(doc_word_freq)))

print('build adjacency_matrix in train dataset and test dataset')
for i in tqdm(range(len(shuffle_doc_words_list))):
    doc_words = shuffle_doc_words_list[i]
    doc_word_set = set()
    for word in doc_words:
        if word in doc_word_set:
            continue
        j = word_id_map[word]
        key = str(i) + ',' + str(j)
        freq = doc_word_freq[key]
        if i < train_size:
            rows.append(i)
        else:
            rows.append(i + vocab_size)
        cols.append(train_size + j)
        idf = log(1.0 * len(shuffle_doc_words_list) /
                  word_doc_freq[vocab[j]])
        weight.append(freq * idf)
        doc_word_set.add(word)

node_size = train_size + vocab_size + test_size
adj = sp.csr_matrix(
    (weight, (rows, cols)), shape=(node_size, node_size))

# dump objects
f = open("./data/graph/ind.{}.x".format(dataset), 'wb')
pkl.dump(x, f)
f.close()

f = open("./data/graph/ind.{}.y".format(dataset), 'wb')
pkl.dump(y, f)
f.close()

f = open("./data/graph/ind.{}.tx".format(dataset), 'wb')
pkl.dump(tx, f)
f.close()

f = open("./data/graph/ind.{}.ty".format(dataset), 'wb')
pkl.dump(ty, f)
f.close()

f = open("./data/graph/ind.{}.allx".format(dataset), 'wb')
pkl.dump(allx, f)
f.close()

f = open("./data/graph/ind.{}.ally".format(dataset), 'wb')
pkl.dump(ally, f)
f.close()

f = open("./data/graph/ind.{}.adj".format(dataset), 'wb')
pkl.dump(adj, f)
f.close()

