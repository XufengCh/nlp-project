import io
import os
import re

import jieba
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf


def load_dataset(path):
    data = []
    with io.open(path, 'r', encoding='UTF-8') as f:
        for line in f:
            line = line.strip('\n').rstrip().strip()
            if line != '':
                data.append([int(item) for item in line.split(' ')])
    assert len(data)
    return data


def store_dataset(dataset, path):
    with io.open(path, 'w', encoding='UTF-8') as f:
        for line in dataset:
            f.write('{}\n'.format(' '.join(map(str, line))))


def text_to_nums(tensor, word2index):
    return [word2index[w] for w in tensor if w in word2index]


def create_dataset_from_raw(raw_data_path, train_enc_path, train_dec_path, test_enc_path, test_dec_path,
                            num_examples, word2index):
    """
    create train_enc, train_dec, test_enc, test_dec from raw_data
    """
    print('Creating datasets from raw data: {}\t...'.format(raw_data_path))
    # word_pairs = []
    enc = []
    dec = []
    try:
        file = io.open(raw_data_path, encoding='UTF-8')
        match_en = r'[a-zA-Z]+'
        for line in file:
            line = line.strip('\n')
            if re.match(match_en, line):
                # ignore the case
                pass
            else:
                line = line.split('\t')
                # add <start> and <end> tags
                question = ['<start>']
                answer = ['<start>']

                # 豆瓣已分好词
                # question += jieba.cut(line[0].rstrip().strip())
                # answer += jieba.cut(line[1].rstrip().strip())
                question += line[0].rstrip().strip().split(' ')
                answer += line[1].rstrip().strip().split(' ')
                question.append('<end>')
                answer.append('<end>')
                # word_pairs.append([question, answer])
                # 过滤太长的对话
                if len(question) < 27 and len(answer) < 27:
                    enc.append(question)
                    dec.append(answer)
        file.close()
    except Exception as e:
        print('Can not open raw data file.\nExit.')
        print('Exception:\t' + e)
        exit()
    # limit the number of examples
    # word_pairs = word_pairs[:num_examples]
    # enc, dec = zip(*word_pairs)
    enc = enc[:num_examples]
    dec = dec[:num_examples]

    # word to index
    enc = [text_to_nums(line, word2index) for line in enc]
    dec = [text_to_nums(line, word2index) for line in dec]

    # padding
    enc = tf.keras.preprocessing.sequence.pad_sequences(enc, padding='post', maxlen=27)
    dec = tf.keras.preprocessing.sequence.pad_sequences(dec, padding='post', maxlen=27)

    train_enc, test_enc, train_dec, test_dec = train_test_split(enc, dec, test_size=0.2)
    store_dataset(train_enc, train_enc_path)
    store_dataset(train_dec, train_dec_path)
    store_dataset(test_enc, test_enc_path)
    store_dataset(test_dec, test_dec_path)
    print('Datasets created...')
    return train_enc, train_dec, test_enc, test_dec


def prepare_data(train_enc_path, train_dec_path, test_enc_path, test_dec_path, raw_data_path,
                 word2index, num_examples=None):
    train_enc = []
    train_dec = []
    test_enc = []
    test_dec = []
    if os.path.exists(train_enc_path) and os.path.exists(train_dec_path) and \
            os.path.exists(test_enc_path) and os.path.exists(test_dec_path):
        train_enc = load_dataset(train_enc_path)
        train_dec = load_dataset(train_dec_path)
        test_enc = load_dataset(test_enc_path)
        test_dec = load_dataset(test_dec_path)
    elif os.path.exists(raw_data_path):
        train_enc, train_dec, test_enc, test_dec = \
            create_dataset_from_raw(raw_data_path, train_enc_path, train_dec_path, test_enc_path, test_dec_path,
                                    num_examples, word2index)
    else:
        print('No datasets available. Exit. ')
        exit()

    return train_enc, train_dec, test_enc, test_dec
