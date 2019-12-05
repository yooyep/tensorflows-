#%% 加载tf_logistic的模型
# -*- coding: utf-8 -*-

import tensorflow as tf
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('Model/model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('Model/'))
    graph = tf.get_default_graph()
    x_input = graph.get_tensor_by_name("xs:0")
    w = graph.get_tensor_by_name("w:0")
    
    predict = graph.get_tensor_by_name("predict:0")
    print(sess.run([predict],feed_dict={x_input:[[1,2]]}))
    print('w:{}'.format(sess.run(w)))
    
