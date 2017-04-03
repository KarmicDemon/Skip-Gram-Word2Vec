from collections import Counter
from download_text8 import download_text8_data
from random import random, sample

import numpy as np
import time
import tensorflow as tf
import utils


### Relevant Functions
def get_batches(words, batch_size, text_margin_size=5):
    n_batches = len(words) // batch_size

    words = words[:n_batches * batch_size]

    for i in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[i:i + batch_size]

        for j in range(len(batch)):
            batch_x = batch[j]
            batch_y = get_target(batch, j, text_margin_size)

            x.extend([batch_x] * len(batch_y))
            y.extend(batch_y)

            yield x, y

def get_target(words, index, text_margin_size = 5):
    R = np.random.randint(1, text_margin_size + 1)

    start = index - R if (index - R) > 0 else 0
    stop = index + R

    target_words = set(words[start:index] + words[index + 1: stop + 1])

    return list(target_words)


### Read in Wikipedia Text8 Data
download_text8_data()
text8_data_file = 'data\\text8'
with open(text8_data_file) as f:
    text = f.read()

### Preprocess Text Data
print('Preprocessing Text8 Dataset')
words = utils.preprocess(text)

vocab_to_int, int_to_vocab = utils.create_lookup_tables(words)
int_words = [vocab_to_int[word] for word in words]

### Mikolov Subsampling to remove Stopwords based on
### http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
print('Subsampling')
t = 1e-5

word_counts = Counter(int_words)
total_count = len(int_words)

word_freq = { word: count/total_count for word, count in word_counts.items() }
drop_probs = { word: 1 - np.sqrt(t / word_freq[word]) \
    for word in word_counts }

train_words = [word for word in int_words if drop_probs[word] < random()]

# define hyperparameters
epochs = 10
batch_size = 1000
window_size = 10

n_embedding = 200
n_sampled = 100
n_vocab = len(int_to_vocab)

valid_size = 16
valid_window = 100

### Building the Graph
print('Building RNN Graph')

train_graph = tf.Graph()
with train_graph.as_default():
    inputs = tf.placeholder(tf.int32, [None], name = 'inputs')
    labels = tf.placeholder(tf.int32, [None, None], name = 'labels')

    embedding = tf.Variable(tf.random_uniform((n_vocab, n_embedding), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, inputs)

    softmax_w = tf.Variable(tf.truncated_normal((n_vocab, n_embedding),
        stddev = 0.1))
    softmax_b = tf.Variable(tf.zeros(n_vocab))

    loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b, labels, embed,
        n_sampled, n_vocab)
    cost = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    valid_examples = np.array(sample(range(valid_window), valid_size // 2))
    valid_examples = np.append(valid_examples, \
        sample(range(1000, 1000 + valid_window), valid_size // 2))
    valid_dataset = tf.constant(valid_examples, dtype = tf.int32)


    ## cosisne distance based on word2vec implementation by Thushan Ganegedara
    norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims = True))
    normalized_embedding = embedding / norm
    valid_embedding = tf.nn.embedding_lookup(normalized_embedding, \
        valid_dataset)
    similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding))

    saver = tf.train.Saver()


### Train RNN
with tf.Session(graph = train_graph) as s:
    _iter = 1
    loss = 0
    s.run(tf.global_variables_initializer())

    for e in range(1, epochs + 1):
        batches = get_batches(train_words, batch_size, window_size)
        for x, y in batches:
            feed = {
                inputs : x,
                labels : np.array(y)[:, None]
            }

            train_loss = s.run([cost, optimizer], feed_dict = feed)
            loss += train_loss[0]

            if _iter % 100 == 0:
                print("Epoch {}/{}".format(e, epochs),
                    "Iteration: {}".format(_iter),
                    "Avg. Training loss: {:.4f}".format(loss/100))
                loss = 0

            if _iter % 1000 == 0:
                sim = similarity.eval()

                for _ in range(valid_size):
                    valid_word = int_to_vocab[valid_examples[_]]
                    nearest = (-sim[_, :]).argsort()[1:9]
                    log = 'Nearest to %s' % valid_word
                    for k in range(8):
                        close_word = int_to_vocab[nearest[k]]
                        log = '%s %s' % (log, close_word)
                    print(log)

            _iter += 1

    path = saver.save(s, 'checkpoints/text8.ckpt')
    embed_mat = s.run(normalized_embedding)


''' Visualize Word Vector Code
import matplotlib.pyplot as plt
from sklearn import TSNE

viz_words = 500
tsne = TSNE()
embed_tsne = TSNE().fit_transform(embed_mat[:viz_words, :])

fig, ax = plt.subplots(figsize = (14, 14))
for _ in range(viz_words):
    plt.scatter(*embed_tsne[_, :], color = 'steelblue')
    plt.annotate(int_to_vocab[_], (embed_tsne[_, 0], embed_tsne[_, 1]), \
        alpha = 0.7)
'''
