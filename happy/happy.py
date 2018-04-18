import tensorflow as tf
import csv

features_for_training = []
labels_for_training = []
features_for_testing = []
labels_for_testing = []

# Read datasets from CSV
with open('2015.csv', newline='') as f:
    reader = csv.reader(f)
    next(reader, None)  # skip the headers
    for row in reader:
        features_for_training.append(row[5:])
        labels_for_training.append(row[3])

with open('2016.csv', newline='') as f:
    reader = csv.reader(f)
    next(reader, None)  # skip the headers
    for row in reader:
        features_for_training.append(row[6:])
        labels_for_training.append(row[3])

with open('2017.csv', newline='') as f:
    reader = csv.reader(f)
    next(reader, None)  # skip the headers
    for row in reader:
        features_for_testing.append(row[5:])
        labels_for_testing.append(row[2])

# Convert data to float
features_for_training = [[float(item) for item in row]
                         for row in features_for_training]
labels_for_training = [float(item) for item in labels_for_training]

features_for_testing = [[float(item) for item in row]
                        for row in features_for_testing]
labels_for_testing = [float(item) for item in labels_for_testing]

# Check arrays shapes
# print(np.array(features_for_training).shape)
# print(np.array(labels_for_training).shape)
# print(np.array(features_for_testing).shape)
# print(np.array(labels_for_testing).shape)

inputs = tf.placeholder(shape=(None, 7),
                        dtype=tf.float32)
outputs = tf.placeholder(shape=(None),
                         dtype=tf.float32)

# weights0 = tf.Variable(initial_value=tf.random_normal(shape=[7, 70]),
#                        dtype=tf.float32)
# weights1 = tf.Variable(initial_value=tf.random_normal(shape=[70, 70]),
#                        dtype=tf.float32)
# weights2 = tf.Variable(initial_value=tf.random_normal(shape=[70, 1]),
#                        dtype=tf.float32)
weights0 = tf.Variable(initial_value=tf.random_uniform(shape=[7, 15],
                                                       minval=0.1,
                                                       maxval=0.3),
                       dtype=tf.float32)
weights1 = tf.Variable(initial_value=tf.random_uniform(shape=[15, 15],
                                                       minval=0.1,
                                                       maxval=0.3),
                       dtype=tf.float32)
weights2 = tf.Variable(initial_value=tf.random_uniform(shape=[15, 15],
                                                       minval=0.1,
                                                       maxval=0.3),
                       dtype=tf.float32)
weights3 = tf.Variable(initial_value=tf.random_uniform(shape=[15, 1],
                                                       minval=0.1,
                                                       maxval=0.3),
                       dtype=tf.float32)

layer1_in = tf.matmul(inputs, weights0)
layer1_out = tf.nn.sigmoid(layer1_in)

layer2_in = tf.matmul(layer1_out, weights1)
layer2_out = tf.nn.relu(layer2_in)

layer3_in = tf.matmul(layer2_out, weights2)
layer3_out = tf.nn.relu(layer3_in)

layer_res_in = tf.matmul(layer3_out, weights3)
layer_res_out = tf.nn.relu(layer_res_in)

# error = tf.losses.mean_squared_error(labels=outputs, predictions=layer_res_out)
error = abs(tf.reduce_mean(outputs - layer_res_out))

train_op = tf.train. \
    AdamOptimizer(learning_rate=0.00005). \
    minimize(error)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter("output", sess.graph)

    er = 50
    step = 0

    while er > 0.000001:
        step += 1
        _, er = sess.run(fetches=[train_op, error],
                         feed_dict={inputs: features_for_training,
                                    outputs: labels_for_training})
        print(step, ": ", er)

    writer.close()

    test = sess.run(fetches=layer_res_out,
                    feed_dict={inputs: [[1.28455626964569, 1.38436901569366, 0.606041550636292, 0.437454283237457, 0.201964423060417, 0.119282886385918, 1.78489255905151]]})
    print(test)
