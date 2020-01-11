from configparser import ConfigParser
from gensim.models import Word2Vec
import sys
import os
import time
import logging
import jieba
import datetime
import tensorflow as tf
import numpy as np

from data_util import prepare_data
from seq2seq_model import Encoder, Decoder

gConfig = {}
max_length_targ = 27
max_length_inp = 27

# for word2vec model
vocab_size = 0
index2word = []
embedding_matrix = []
word2index = {}


def load_word2vec_model(path):
    global vocab_size, index2word, embedding_matrix, word2index
    try:
        # load pretrained word2vec model
        word2vec = Word2Vec.load(path)
        vocab_size = len(word2vec.wv.vocab)
        index2word = word2vec.wv.index2word
        embedding_matrix = word2vec.wv.syn0
        word2index = {}
        for i in range(vocab_size):
            word2index[index2word[i]] = i
    except:
        print('Can not Word2Vec model. Exit. ')
        exit()


def load_config(config_file='seq2seq.ini'):
    parser = ConfigParser()
    parser.read(config_file)
    # get the ints, floats and strings
    _conf_ints = [(key, int(value)) for key, value in parser.items('ints')]
    _conf_strings = [(key, str(value)) for key, value in parser.items('strings')]

    res = {}
    for conf in [_conf_ints, _conf_strings]:
        for (key, value) in conf:
            res[key] = value
    return res


def max_length(tensor):
    return max(len(t) for t in tensor)


def loss_function(real, pred, loss_object):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def train():
    global gConfig, max_length_targ, max_length_inp
    # optimizer and loss function
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none'
    )

    # load_word2vec_model(gConfig['word2vec_model'])

    # prepare dataset
    input_tensor_train, target_tensor_train, input_tensor_test, target_tensor_test = prepare_data(gConfig['train_enc'],
                                                                                                  gConfig['train_dec'],
                                                                                                  gConfig['test_enc'],
                                                                                                  gConfig['test_dec'],
                                                                                                  gConfig['raw_data'],
                                                                                                  word2index,
                                                                                                  100000)

    # max_length_targ = max(max_length(target_tensor_train), max_length(target_tensor_test))
    # max_length_inp = max(max_length(input_tensor_train), max_length(input_tensor_test))

    BUFFER_SIZE = len(input_tensor_train)
    BATCH_SIZE = gConfig['batch_size']
    steps_per_epoch = BUFFER_SIZE // BATCH_SIZE
    embedding_dim = 300
    units = gConfig['layer_size']

    train_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)

    test_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_test, target_tensor_test)).shuffle(
        len(input_tensor_test))
    test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=True)

    print("\ndatasets created...\n")

    # create model
    print('<<<---------- creating models ---------->>>\n')
    encoder = Encoder(vocab_size, embedding_dim, units, BATCH_SIZE, embedding_matrix)
    decoder = Decoder(vocab_size, embedding_dim, units, BATCH_SIZE, embedding_matrix)
    print('<<<---------- finish creating models ---------->>>\n')

    # checkpoint
    checkpoint_dir = gConfig['checkpoint_dir']
    checkpoint_prefix = os.path.join(checkpoint_dir, 'seq2seq_ckpt')
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)

    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

    @tf.function
    def train_step(inp_train, targ_train, enc_hidden_train):
        # print('run train_step')
        loss = 0

        with tf.GradientTape() as tape:
            # print('start tape')
            enc_output, enc_hidden_train = encoder(inp_train, enc_hidden_train)
            # print('get encoder output')
            dec_hidden = enc_hidden_train

            dec_input = tf.expand_dims([word2index['<start>']] * BATCH_SIZE, 1)

            # 教师强制 - 将目标词作为下一个输入
            for t in range(1, targ_train.shape[1]):
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
                # print('decoder input {}'.format(t))
                loss += loss_function(targ_train[:, t], predictions, loss_object)

                # 使用教师强制
                dec_input = tf.expand_dims(targ_train[:, t], 1)

        batch_loss_train = (loss / int(targ_train.shape[1]))
        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        train_loss(batch_loss_train)
        return batch_loss_train

    def test_step(inp_test, targ_test, enc_hidden_test):
        loss = 0

        enc_output, enc_hidden_test = encoder(inp_test, enc_hidden_test)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([word2index['<start>']] * BATCH_SIZE, 1)

        # 教师强制 - 将目标词作为下一个输入
        for t in range(1, targ_test.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(targ_test[:, t], predictions, loss_object)

            dec_input = tf.expand_dims(targ_test[:, t], 1)

        batch_test_loss = loss / int(targ_test.shape[1])
        # test_loss(batch_test_loss)
        return batch_test_loss

    # for tensorboard
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = './logs/' + current_time + '/train'
    test_log_dir = './logs/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    epochs = gConfig['epochs']

    print('\n<<<----------    start training...    ---------->>>\n')
    for epoch in range(epochs):
        start = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(train_dataset.take(steps_per_epoch)):
            # print('input shape :')
            # print(inp.shape)
            # print('targ shape :')
            # print(targ.shape)
            batch_loss = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss

            with train_summary_writer.as_default():
                tf.summary.scalar('train_loss', train_loss.result(), step=(int(batch + epoch * steps_per_epoch)))
            train_loss.reset_states()

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))

        test_epoch_loss = 0
        for (batch, (inp, targ)) in enumerate(test_dataset.take(int(len(input_tensor_test) / BATCH_SIZE))):
            test_epoch_loss += test_step(inp, targ, enc_hidden)

        test_loss(test_epoch_loss / (len(input_tensor_test) / BATCH_SIZE))
        with test_summary_writer.as_default():
            tf.summary.scalar('test_loss_epoch', test_loss.result(), step=epoch)
        test_loss.reset_states()

        checkpoint.save(file_prefix=checkpoint_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


def evaluate(sentence, encoder, decoder):
    global max_length_targ, max_length_inp
    # init attention plot
    attention_plot = np.zeros((max_length_targ, max_length_inp))

    # inputs = [word2index[word] for word in sentence]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([sentence], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, gConfig['layer_size']))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([word2index['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

        attention_weights = tf.reshape(attention_weights, (-1,))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        if index2word[predicted_id] == '<end>':
            return result, sentence, attention_plot

        result += index2word[predicted_id]
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot


def predict():
    # load model
    global vocab_size, embedding_matrix
    try:
        BATCH_SIZE = gConfig['batch_size']
        embedding_dim = 300
        units = gConfig['layer_size']
        encoder = Encoder(vocab_size, embedding_dim, units, BATCH_SIZE, embedding_matrix)
        decoder = Decoder(vocab_size, embedding_dim, units, BATCH_SIZE, embedding_matrix)

        optimizer = tf.keras.optimizers.Adam()

        checkpoint_dir = gConfig['checkpoint_dir']
        checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                         encoder=encoder,
                                         decoder=decoder)

        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    except:
        print('Restore model failed. \nExit. ')
        exit()

    print('Enter \'exit\' to quit:\n')

    while True:
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline().strip('\n').rstrip().strip()

        if sentence == 'exit':
            # exit the program
            break
        else:
            tokens = jieba.cut(sentence)

            # words -> index
            sentence = []
            is_valid = True
            for w in tokens:
                if w in word2index:
                    sentence.append(word2index[w])
                else:
                    # words not in vocab
                    is_valid = False
                    print('我不知道你在说什么\n')
                    break

            if is_valid:
                answer, _, _ = evaluate(sentence, encoder, decoder)
                print('{}\n'.format(answer))


if __name__ == '__main__':
    if len(sys.argv) - 1:
        gConfig = load_config(sys.argv[1])
    else:
        gConfig = load_config()

    print('\n>> Mode : %s\n' % (gConfig['mode']))

    # load word2vec model
    load_word2vec_model(gConfig['word2vec_model'])

    if gConfig['mode'] == 'train':
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(sys.argv[0])
        train()
    elif gConfig['mode'] == 'predict':
        predict()
