import tensorflow as tf

input_data=[[1,5,3,7,8,10,12],[5, 8, 10, 3, 9, 7,1]]
label_data=[[0,0,0,1,0], [1, 0 ,0, 0, 0]]

INPUT_SIZE=7
HIDDEN1_SIZE=10
HIDDEN2_SIZE=8
CLASSES=5
Learning_Rate=0.05

# shape must be matched to data dimension
x=tf.placeholder(tf.float32, shape=[None, INPUT_SIZE], name='x') #shape=[batchSize, dimension]
y_=tf.placeholder(tf.float32, shape=[None, CLASSES], name='y_')

tensor_map={x:input_data, y_:label_data}

# Building Model
W_h1 =tf.Variable(tf.truncated_normal(shape=[INPUT_SIZE, HIDDEN1_SIZE]), dtype=tf.float32, name='W_h1') # truncated_normal: Outputs random values from a normal distribution
b_h1 = tf.Variable(tf.zeros(shape=[HIDDEN1_SIZE]), dtype=tf.float32, name='b_h1')

W_h2 =tf.Variable(tf.truncated_normal(shape=[HIDDEN1_SIZE, HIDDEN2_SIZE]), dtype=tf.float32, name='W_h2')
b_h2 = tf.Variable(tf.zeros(shape=[HIDDEN2_SIZE]), dtype=tf.float32, name='b_h2')

W_o = tf.Variable(tf.truncated_normal(shape=[HIDDEN2_SIZE, CLASSES]), dtype=tf.float32, name='W_o')
b_o = tf.Variable(tf.zeros(shape=[CLASSES]), dtype=tf.float32, name='b_o')

param_list=[W_h1, b_h1, W_h2, b_h2, W_o, b_o]
saver = tf.train.Saver(param_list)

hidden1=tf.sigmoid(tf.matmul(x, W_h1) +b_h1, name='hidden1')
hidden2=tf.sigmoid(tf.matmul(hidden1,W_h2) + b_h2, name='hidden2')
y=tf.sigmoid(tf.matmul(hidden2, W_o) +b_o, name='y')

sess = tf.Session()
# sess.run(tf.initialize_all_variables())
saver.restore(sess, './tensorflow_checkpoint.ckpt')
result=sess.run(y, tensor_map)
print(result)
