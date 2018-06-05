# TODO почистить данные для тренировки
import tensorflow as tf
import numpy as np
# tf.enable_eager_execution()


def encode_letter(letter: str) -> float:
    alphabet = "аәбвгғдеёжзийкқлмнңоөпрстуұүфхһцчшщъыіьэюя"
    codes = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
             12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0,
             23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0,
             34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0]

    return codes[alphabet.find(letter)]


def pad_to_dense(M):
    """Appends the minimal required amount of zeroes at the end of each
     array in the jagged array `M`, such that `M` looses its jagedness.
     https://stackoverflow.com/questions/37676539/
     numpy-padding-matrix-of-different-row-size
     """
    maxlen = max(len(r) for r in M)

    Z = np.zeros((len(M), maxlen))
    for enu, row in enumerate(M):
        Z[enu, :len(row)] += row
    return Z


# загрузка и преобразование данных
# читаем все строки из файла с лексиконом
with open("data/gen-data") as f:
    lines = f.readlines()

lemmas = []
words = []
tags_list = []

# NB тут можно сделать shuffle
for line in lines:
    # делим каждую строку на лемму, теги и слово. Резделитель - символ ':'
    sep_ind_1 = line.find(':')
    sep_ind_2 = line.rfind(':')
    lemma = line[:sep_ind_1]
    tags = line[sep_ind_1 + 1:sep_ind_2]
    word = line[sep_ind_2:]

    # кодируем буквы слова
    coded_lemma = []
    coded_word = []
    for letter in lemma:
        coded_lemma.append(encode_letter(letter))
    for letter in word:
        coded_word.append(encode_letter(letter))

    lemmas.append(coded_lemma)
    words.append(coded_word)

    # "выуживаем" и кодируем теги
    tag_codes = []
    if "possession" in tags:
        if "<px1sg>" in tags:
            tag_codes += [1, 0, 0, 0, 0, 0, 0, 0, 0]
        if "<px1pl>" in tags:
            tag_codes += [0, 1, 0, 0, 0, 0, 0, 0, 0]
        if "<px3sg>" in tags:
            tag_codes += [0, 0, 1, 0, 0, 0, 0, 0, 0]
        if "<px3pl>" in tags:
            tag_codes += [0, 0, 0, 1, 0, 0, 0, 0, 0]
        if "<px3sp>" in tags:
            tag_codes += [0, 0, 0, 0, 1, 0, 0, 0, 0]
        if "<px2sg>" in tags:
            tag_codes += [0, 0, 0, 0, 0, 1, 0, 0, 0]
        if "<px2pl>" in tags:
            tag_codes += [0, 0, 0, 0, 0, 0, 1, 0, 0]
        if "<px2sp>" in tags:
            tag_codes += [0, 0, 0, 0, 0, 0, 0, 1, 0]
        if "<px2sp_2>" in tags:
            tag_codes += [0, 0, 0, 0, 0, 0, 0, 0, 1]
    else:
        tag_codes += [0, 0, 0, 0, 0, 0, 0, 0, 0]

    if "plurality" in tags:
        if "<sg>" in tags:
            tag_codes += [1, 0, 0]
        if "<pl>" in tags:
            tag_codes += [0, 1, 0]
        if "<sp>" in tags:
            tag_codes += [0, 0, 1]
    else:
        tag_codes += [0, 0, 0]

    if "case" in tags:
        if "<nom>" in tags:
            tag_codes += [1, 0, 0, 0, 0, 0, 0]
        if "<gen>" in tags:
            tag_codes += [0, 1, 0, 0, 0, 0, 0]
        if "<dat>" in tags:
            tag_codes += [0, 0, 1, 0, 0, 0, 0]
        if "<acc>" in tags:
            tag_codes += [0, 0, 0, 1, 0, 0, 0]
        if "<loc>" in tags:
            tag_codes += [0, 0, 0, 0, 1, 0, 0]
        if "<abl>" in tags:
            tag_codes += [0, 0, 0, 0, 0, 1, 0]
        if "<ins>" in tags:
            tag_codes += [0, 0, 0, 0, 0, 0, 1]
    else:
        tag_codes += [0, 0, 0, 0, 0, 0, 0]

    if "person" in tags:
        if "<p1>" in tags:
            tag_codes += [1, 0, 0, 0]
        if "<p2>" in tags:
            tag_codes += [0, 1, 0, 0]
        if "<p2_2>" in tags:
            tag_codes += [0, 0, 1, 0]
        if "<p3>" in tags:
            tag_codes += [0, 0, 0, 1]
    else:
        tag_codes += [0, 0, 0, 0]

    tags_list.append(tag_codes)

# NB здесь мог быть ваш embedding

# "выравниваем" массивы слов
lemmas = pad_to_dense(lemmas)
words = pad_to_dense(words)

features = []
labels = []

for i in range(len(lemmas)):
    features.append(list(lemmas[i]) + list(tags_list[i]))
    labels.append(words[i])

num_epochs = 10
batch_size = 10
train_test_ratio = 0.8

ind = int(len(features) * train_test_ratio)
# леммы и векторы грамматических характеристик
features_for_training = features[:ind]
# слова
labels_for_training = labels[:ind]

# леммы и векторы грамматических характеристик
features_for_testing = features[ind:]
# слова
labels_for_testing = labels[ind:]

input_dim = len(features_for_training[0])
output_dim = len(labels_for_training[0])
hidden_dim = 200

# начинается модель
inputs_placeholder = tf.placeholder(shape=[None, input_dim], dtype=tf.float32)
outputs_placeholder = tf.placeholder(shape=[None, output_dim],
                                     dtype=tf.float32)

# Preparing nn parameters (weights) using tf Variables
weights_0_1 = tf.Variable(
    initial_value=tf.random_uniform(shape=[input_dim, hidden_dim],
                                    maxval=0.01),
    dtype=tf.float32)

biases_0_1 = tf.Variable(
    initial_value=tf.zeros(shape=[hidden_dim], dtype=tf.float32),
    dtype=tf.float32)

# ==========

weights_1_2 = tf.Variable(
    initial_value=tf.random_uniform(shape=[hidden_dim, hidden_dim],
                                    maxval=0.01),
    dtype=tf.float32)

biases_1_2 = tf.Variable(
    initial_value=tf.zeros(shape=[hidden_dim], dtype=tf.float32),
    dtype=tf.float32)

# ==========

weights_2_3 = tf.Variable(
    initial_value=tf.random_uniform(shape=[hidden_dim, hidden_dim],
                                    maxval=0.01),
    dtype=tf.float32)

biases_2_3 = tf.Variable(
    initial_value=tf.zeros(shape=[hidden_dim], dtype=tf.float32),
    dtype=tf.float32)

# ==========

weights_3_out = tf.Variable(
    initial_value=tf.random_uniform(shape=[hidden_dim, output_dim],
                                    maxval=0.01),
    dtype=tf.float32)

biases_3_out = tf.Variable(
    initial_value=tf.zeros(shape=[output_dim], dtype=tf.float32),
    dtype=tf.float32)
# ==========

# Create layers
layer1_in = tf.matmul(inputs_placeholder, weights_0_1)
layer1_out = tf.nn.sigmoid(layer1_in + biases_0_1)

layer2_in = tf.matmul(layer1_out, weights_1_2)
layer2_out = tf.nn.sigmoid(layer2_in + biases_1_2)

layer3_in = tf.matmul(layer2_out, weights_2_3)
layer3_out = tf.nn.sigmoid(layer3_in + biases_2_3)

layer_res_in = tf.matmul(layer3_out, weights_3_out)
layer_res_out = tf.nn.relu(layer_res_in + biases_3_out)

loss = tf.losses.mean_squared_error(labels=outputs_placeholder,
                                    predictions=layer_res_out)

# Minimizing the prediction error using gradient descent optimizer
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.05). \
    minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # training
    for epoch in range(num_epochs * 10000):
        # поделить на batches
        _, current_loss = sess.run(
            fetches=[train_op, loss],
            feed_dict={inputs_placeholder: features_for_training,
                       outputs_placeholder: labels_for_training})
        print("Epoch:", epoch, "current_loss:", current_loss)

    print("training_loss:", current_loss)

    # testing
    testing_loss = sess.run(
        fetches=loss,
        feed_dict={inputs_placeholder: features_for_testing,
                   outputs_placeholder: labels_for_testing})

    print("testing_loss:", testing_loss)
