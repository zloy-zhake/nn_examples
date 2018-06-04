# TODO почистить данные для тренировки
import tensorflow as tf
import numpy as np
# tf.enable_eager_execution()


def encode_letter(letter: str) -> float:
    alphabet = "аәбвгғдеёжзийкқлмнңоөпрстуұүфхһцчшщъыіьэюя"
    codes = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
             12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0,
             23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0,
             34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0]

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


num_epochs = 10
batch_size = 10
train_test_ratio = 0.8
# результат: 1х23
output_dim = 23
embedding_dim = 50
history_dim = 50

# читаем все строки из файла с лексиконом
with open("data/noun-lexicon") as f:
    lines = f.readlines()

features = []
labels = []

# NB тут можно сделать shuffle
for line in lines:
    # делим каждую строку на слово и теги. Резделитель - первый символ ':'
    sep_ind = line.find(':')
    word = line[:sep_ind]
    tags = line[sep_ind + 1:]

    # кодируем буквы слова и записываем
    coded_word = []
    for letter in word:
        coded_word.append(encode_letter(letter))
    features.append(coded_word)

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

    # записываем теги
    labels.append(tag_codes)

# NB здесь мог быть ваш embedding

# "выравниваем" массив признаков
features = pad_to_dense(features)

# загрузка и преобразование данных
ind = int(len(features) * train_test_ratio)
# слова
features_for_training = features[:ind]
# вектор грамматических характеристик
labels_for_training = labels[:ind]

# слова
features_for_testing = features[ind:]
# вектор грамматических характеристик
labels_for_testing = labels[ind:]

# начинается модель
inputs_placeholder_training = tf.placeholder(
    shape=(len(features_for_training), len(features_for_training[0])),
    dtype=tf.float32)
outputs_placeholder_training = tf.placeholder(shape=(None, output_dim),
                                              dtype=tf.float32)

inputs_placeholder_testing = tf.placeholder(
    shape=(len(features_for_testing), len(features_for_testing[0])),
    dtype=tf.float32)
outputs_placeholder_testing = tf.placeholder(shape=(None, output_dim),
                                             dtype=tf.float32)

history_state = tf.Variable(
    initial_value=tf.zeros(shape=[1, history_dim], dtype=tf.float32),
    dtype=None)

history_weights = tf.Variable(
    initial_value=tf.eye(num_rows=history_dim, dtype=tf.float32),
    dtype=None)

RNN_biases = tf.Variable(
    initial_value=tf.zeros(shape=[history_dim], dtype=tf.float32),
    dtype=None)

input_weights = tf.Variable(
    initial_value=tf.random_uniform(shape=[embedding_dim, history_dim],
                                    maxval=0.1, dtype=tf.float32),
    dtype=None)

output_weights = tf.Variable(
    initial_value=tf.random_uniform(shape=[history_dim, output_dim],
                                    maxval=0.1, dtype=tf.float32),
    dtype=None)

output_biases = tf.Variable(
    initial_value=tf.zeros(shape=[output_dim], dtype=tf.float32),
    dtype=None)


# Так как результат обработки вычисляется в конце,
# RNN_step возвращает только новое значение history_state
def RNN_step(input_item: tf.Tensor, history: tf.Tensor) -> tf.Tensor:
    # FIXME
    t = tf.expand_dims(tf.tile(tf.reshape(input_item, [1]), tf.constant([50])), 0)
    input_x_weights = tf.matmul(t, input_weights)
    history_x_weights = tf.matmul(history, history_weights)
    # NB можно попробовать конкатенацию вместо сложения
    input_history_sum = tf.add(input_x_weights, history_x_weights)
    # NB можно попробовать другие функции активации
    return tf.nn.sigmoid(input_history_sum + RNN_biases)


inp_unst = tf.unstack(inputs_placeholder_training, axis=0)
for item in inp_unst:
    lin_unst = tf.unstack(item, axis=0)
    for letter_code in lin_unst:
        history_state = RNN_step(input_item=letter_code, history=history_state)

history_x_output_weights = tf.matmul(history_state, output_weights)
RNN_result = tf.nn.sigmoid(history_x_output_weights + output_biases)

# ==========
# for testing
history_state_testing = history_state = tf.Variable(
    initial_value=tf.zeros(shape=[1, history_dim], dtype=tf.float32),
    dtype=None)

inp_unst_testing = tf.unstack(inputs_placeholder_testing, axis=0)
for item in inp_unst_testing:
    lin_unst = tf.unstack(item, axis=0)
    for letter_code in lin_unst:
        history_state_testing = RNN_step(input_item=letter_code,
                                         history=history_state_testing)

history_x_output_weights_testing = tf.matmul(history_state_testing,
                                             output_weights)
RNN_result_testing = \
    tf.nn.sigmoid(history_x_output_weights_testing + output_biases)
# ==========

# error_training = tf.reduce_mean(outputs_placeholder_training - RNN_result)
error_training =\
    tf.losses.mean_squared_error(outputs_placeholder_training, RNN_result)
# error_testing =\
    # tf.reduce_mean(outputs_placeholder_testing - RNN_result_testing)
error_testing =\
    tf.losses.mean_squared_error(outputs_placeholder_testing,
                                 RNN_result_testing)

train_op = tf.train.AdagradOptimizer(learning_rate=0.05).\
    minimize(error_training)

with tf.Session() as sess:
    writer = tf.summary.FileWriter("output", sess.graph)
    sess.run(tf.global_variables_initializer())

    # print(sess.run(result, feed_dict={inputs_placeholder: features_for_training,
                   # outputs_placeholder: labels_for_training}).shape)
    # input()
    
    # training
    for epoch in range(num_epochs):
        # поделить на batches
        _, current_error = sess.run(
            fetches=[train_op, error_training],
            feed_dict={inputs_placeholder_training: features_for_training,
                       outputs_placeholder_training: labels_for_training})
        print("Epoch:", epoch, "current_error:", current_error)

    print("training_error:", current_error)

    # testing
    testing_error = sess.run(
        fetches=error_testing,
        feed_dict={inputs_placeholder_testing: features_for_testing,
                   outputs_placeholder_testing: labels_for_testing})

    print("testing_error:", testing_error)

    writer.close()    
