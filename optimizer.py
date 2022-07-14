import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
flags = tf.app.flags
FLAGS = flags.FLAGS

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.T / weight.sum(1)).T


class OptimizerAE(object):
    def __init__(self, reconstructions1, reconstructions2,adj,adj2 ,reconstructions3,features,model,global_step,learning_rate, pos_weight, pos_weight2, norm,norm2,se,indices,p):

        # q = sess.run(model.cluster_layer_q, feed_dict=feed_dict)
        # p = target_distribution(q)
        self.cost1 = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=reconstructions1, targets=adj, pos_weight=pos_weight))
        self.cost2 = norm2 * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=reconstructions2, targets=adj2, pos_weight=pos_weight2))
        self.cost3 =tf.losses.mean_squared_error(reconstructions3,features)
        self.cost =self.cost1 +self.cost3+self.cost2

        q = model.cluster_layer_q
        self.kl_loss = tf.reduce_mean(p * tf.log(p / q))
        self.cost_kl = self.cost + self.kl_loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)  # Adam Optimizer
        #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer
        self.opt_op = self.optimizer.minimize(self.cost,global_step=global_step)
        self.optimizer2 = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate2)  # Adam Optimizer
        #self.optimizer2 = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate2)  # Adam Optimizer
        self.opt_op2 = self.optimizer2.minimize(self.cost_kl)
        #self.grads_vars = self.optimizer.compute_gradients(self.cost)

        # self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
        #                                    tf.cast(labels_sub, tf.int32))
        # self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))



class OptimizerVAE(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels


        self.cost1 = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        # Latent loss
        self.log_lik = self.cost1
        self.kl = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) -
                                                                tf.square(tf.exp(model.z_log_std)), 1))
        #self.vin =tf.matmul(tf.matrix_inverse(u*tf.eye(FLAGS.hidden2)+nam*tf.matmul(model.z_mean,tf.transpose(model.z_mean))), tf.matmul(model.z_mean,tf.transpose(model.z_mean)))
        #self.cost2 = (u*nam)/2*tf.trace(self.vin)
        distance = tf.reduce_sum(tf.square(tf.subtract(model.z_mean, tf.expand_dims(model.z_mean, 1))), axis=2)
        d, top_k_indices = tf.nn.top_k(tf.negative(distance), k=100)
        self.cost2 = tf.reduce_mean(tf.negative(d))
        self.cost1 -= self.kl
        self.cost = self.cost1+self.cost2
        #self.cost += norm *self.cost2
        self.opt_op = self.optimizer.minimize(self.cost1)
        self.opt_op2 = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost1)

        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
