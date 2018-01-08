import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import matplotlib.pyplot as plt
import numpy as np
import math

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def lrelu(x, leak=0.2, name="lrelu"):
    """Leaky rectifier.
    Parameters
    ----------
    x : Tensor
        The tensor to apply the nonlinearity to.
    leak : float, optional
        Leakage parameter.
    name : str, optional
        Variable scope to use.
    Returns
    -------
    x : Tensor
        Output of the nonlinearity.
    """
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

class BayesianConvAutoencoder(object):
    def __init__(self, batch_size, mc_samples, constant_prior=False):
        # SIZES
        tf.reset_default_graph()
        
        self.N = mnist.train.num_examples
        self.M = batch_size
        ## Set the number of Monte Carlo samples as a placeholder so that it can be different for training and test
        # self.L =  tf.placeholder(tf.int32)
        self.L = mc_samples
        
        self.constant_prior = constant_prior
        
        ## Batch data placeholders
        with tf.name_scope('input'):
            self.X = tf.placeholder(tf.float32, shape=[None, 784], name='x-input')
            self.Y = tf.placeholder(tf.float32, shape=[None, 784], name='y-input')
            
        with tf.name_scope('input_reshape'):
            self.X_reshaped = tf.reshape(self.X, [-1, 28, 28, 1])
            tf.summary.image('input', self.X_reshaped, 10)
            
        self.phase = tf.placeholder(tf.bool, name='phase')

        self.weights = {}
        
        self.initialize_model_weights()
        
        ## Initialize the session
        self.session = tf.Session()
        
        ## Builds whole computational graph with relevant quantities as part of the class
        self.loss, self.kl, self.ell, self.layer_out = self.get_nelbo()
        
    def add_weights(self, shape, bias_shape, name):
        
        with tf.name_scope(name + "_weights"):
            self.weights[name] = {}
            
            # PRIORS
            if self.constant_prior:
                self.weights[name]['prior_mean'] = tf.constant(0.0, shape=shape, name="prior_mean")
                self.weights[name]['prior_mean_bias'] = tf.constant(0.0, shape=bias_shape, name="prior_mean_bias")
                self.weights[name]['prior_log_var'] = tf.constant(0.0, shape=shape, name="prior_log_var")
                self.weights[name]['prior_log_var_bias'] = tf.constant(0.0, shape=bias_shape, name="prior_log_var_bias")
            else:
                self.weights[name]['prior_mean'] = tf.Variable(tf.zeros(shape), name="prior_mean")
                self.weights[name]['prior_mean_bias'] = tf.Variable(tf.zeros(bias_shape), name="prior_mean_bias")
                self.weights[name]['prior_log_var'] = tf.Variable(tf.zeros(shape), name="prior_log_var")
                self.weights[name]['prior_log_var_bias'] = tf.Variable(tf.zeros(bias_shape), name="prior_log_var_bias")


            #tf.summary.histogram('prior_mean', self.weights[name]['prior_mean'])
            #tf.summary.histogram('prior_logvar', self.weights[name]['prior_log_var'])
        
            # POSTERIORS
            self.weights[name]['post_mean'] = tf.Variable(tf.zeros(shape), name="post_mean")
            self.weights[name]['post_mean_bias'] = tf.Variable(tf.zeros(bias_shape), name="post_mean_bias")
            self.weights[name]['post_log_var'] = tf.Variable(tf.zeros(shape), name="post_log_var")
            self.weights[name]['post_log_var_bias'] = tf.Variable(tf.zeros(bias_shape), name="post_log_var_bias")
            
            #tf.summary.histogram('posterior_mean', self.weights[name]['post_mean'])
            #tf.summary.histogram('posterior_logvar', self.weights[name]['post_log_var'])
            
    def initialize_model_weights(self):        
        # convolution
        self.add_weights([3, 3, 1, 10], [10], 'W_conv1')
        self.add_weights([3, 3, 10, 10], [10], 'W_conv2')
        self.add_weights([3, 3, 10, 10], [10], 'W_conv3')
        
        # "deconvolution"
        self.add_weights([3, 3, 10, 10], [10], 'W_conv4')
        self.add_weights([3, 3, 10, 10], [10], 'W_conv5')
        self.add_weights([3, 3, 1, 10], [1], 'W_conv6')
        
        # increase feature maps while decreasing size
    
    def get_std_norm_samples(self, shape):
        """
        Draws N(0,1) samples of dimension shape.
        """
        return tf.random_normal(shape=shape)

    def sample_from_W(self, layer_name, bias=False):
        """
        Samples from the variational posterior approximation.
        We draw W-samples using the reparameterization trick.
        """
        if bias:
            mean_W = self.weights[layer_name]['post_mean_bias']
            log_var_W = self.weights[layer_name]['post_log_var_bias']
        else:
            mean_W = self.weights[layer_name]['post_mean']
            log_var_W = self.weights[layer_name]['post_log_var']
        
        shape = mean_W.shape
        
        z = self.get_std_norm_samples(shape)
        ## division by 2 to obtain pure standard deviation
        w_from_q = tf.add(tf.multiply(z, tf.exp(log_var_W / 2)), mean_W)
        
        return w_from_q
    
    def feedforward(self):
        """
        Feedforward pass excluding last layer's transfer function.
        intermediate : index of intermediate layer for output generation
        """
        
        # We will generate L output samples
        for i in range(self.L):
            
            # ENCODING
            
            outputs = self.X_reshaped
            
            W = self.sample_from_W('W_conv1')
            W_bias = self.sample_from_W('W_conv1', bias=True)
            
            outputs = tf.add(tf.nn.conv2d(
                outputs, W, strides=[1, 2, 2, 1], padding='SAME'), W_bias)
            
            # tf.summary.histogram('layer_one_activations', outputs)
            # outputs = tf.layers.batch_normalization(outputs, training=self.phase)
            outputs = lrelu(outputs)
            
            W = self.sample_from_W('W_conv2')
            W_bias = self.sample_from_W('W_conv2', bias=True)
            
            outputs = tf.add(tf.nn.conv2d(
                outputs, W, strides=[1, 2, 2, 1], padding='SAME'), W_bias)
            #tf.summary.histogram('layer_two_activations', outputs)
            #outputs = tf.layers.batch_normalization(outputs, training=self.phase)
            outputs = lrelu(outputs)
            
            W = self.sample_from_W('W_conv3')
            W_bias = self.sample_from_W('W_conv3', bias=True)
            
            outputs = tf.add(tf.nn.conv2d(
                outputs, W, strides=[1, 2, 2, 1], padding='SAME'), W_bias)
            #tf.summary.histogram('layer_three_activations', outputs)
            #outputs = tf.layers.batch_normalization(outputs, training=self.phase)
            outputs = lrelu(outputs)
            
            # Latent representation
            z = outputs
            
            # DECODING
            W = self.sample_from_W('W_conv4')
            W_bias = self.sample_from_W('W_conv4', bias=True)
            
            outputs = tf.add(
                tf.nn.conv2d_transpose(
                    outputs, W,
                    tf.stack([tf.shape(self.X_reshaped)[0], 7, 7, 10]),
                    strides=[1, 2, 2, 1], padding='SAME'), W_bias)
            #tf.summary.histogram('layer_four_activations', outputs)
            #outputs = tf.layers.batch_normalization(outputs, training=self.phase)
            outputs = lrelu(outputs)
            
            W = self.sample_from_W('W_conv5')
            W_bias = self.sample_from_W('W_conv5', bias=True)
            
            outputs = tf.add(
                tf.nn.conv2d_transpose(
                    outputs, W,
                    tf.stack([tf.shape(self.X_reshaped)[0], 14, 14, 10]),
                    strides=[1, 2, 2, 1], padding='SAME'), W_bias)
            # tf.summary.histogram('layer_five_activations', outputs)
            #outputs = tf.layers.batch_normalization(outputs, training=self.phase)
            outputs = lrelu(outputs)
            
            W = self.sample_from_W('W_conv6')
            W_bias = self.sample_from_W('W_conv6', bias=True)
            
            outputs = tf.add(
                tf.nn.conv2d_transpose(
                    outputs, W,
                    tf.stack([tf.shape(self.X_reshaped)[0], 28, 28, 1]),
                    strides=[1, 2, 2, 1], padding='SAME'), W_bias)
            #tf.summary.histogram('layer_six_activations', outputs)
            #outputs = tf.layers.batch_normalization(outputs, training=self.phase)
            # outputs = tf.sigmoid(outputs)
            outputs = lrelu(outputs)

            yield tf.contrib.layers.flatten(outputs)

    """
    # Bernoulli Likelihood
    def get_ll(self, target, output):
        return tf.reduce_sum(
            target * tf.log(output + 1e-10) + \
            (1 - target) * tf.log(1 - output + 1e-10),
            reduction_indices=[1]
        )
    """
    
    # Gaussian Likelihood with stddev=1
    def get_ll(self, target, output):
        return tf.reduce_sum(-0.5 * tf.log(2*math.pi) - 0.5 * tf.square(output - target), reduction_indices=[1])

    # def get_ll(self, target, output):
    #     return tf.reduce_sum(tf.square(target - output))
    
    # def get_ll(self, target, output):
    #    return tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output)


    
    def get_ell(self):
        """
        Returns the expected log-likelihood of the lower bound.
        For this we draw L samples from W, compute the log-likelihood for each
        and average the log-likelihoods in the end (expectation approximation).
        The log-likelihoods are returned for a batch of data.
        """
        
        batch_log_p = 0
        cum_output = 0
        inters = []
        
        for output in self.feedforward():
            # y = tf.nn.softmax(tf.matmul(self.X, W_sample[i]) + b)
            # log_p_per_sample = tf.reduce_mean(tf.reduce_sum(self.Y * tf.log(y), reduction_indices=[1]))
            # soft_max_cross_entropy_with_logits is a numerically stable version of cross entropy
            # log_p_per_sample = tf.reduce_mean(-tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=self.Y))
            batch_log_p += self.get_ll(self.Y, output)
            cum_output += output
            
        
        ell = batch_log_p / self.L
        avg_output = cum_output / self.L
        
        return ell, avg_output

    def get_kl(self, layer_name):
        """
        KL[q || p] returns the KL-divergence between the prior p and the variational posterior q.
        :param mq: vector of means for q
        :param log_vq: vector of log-variances for q
        :param mp: vector of means for p
        :param log_vp: vector of log-variances for p
        :return: KL divergence between q and p
        """
        mq = self.weights[layer_name]['post_mean']
        mq_b = self.weights[layer_name]['post_mean_bias']
        log_vq = self.weights[layer_name]['post_log_var']
        log_vq_b = self.weights[layer_name]['post_log_var_bias']
        mp = self.weights[layer_name]['prior_mean']
        mp_b = self.weights[layer_name]['prior_mean_bias']
        log_vp = self.weights[layer_name]['prior_log_var']
        log_vp_b = self.weights[layer_name]['prior_log_var_bias']
        
        kl_normal = 0.5 * tf.reduce_sum(log_vp - log_vq + (tf.pow(mq - mp, 2) / tf.exp(log_vp)) \
                                   + tf.exp(log_vq - log_vp) - 1)
        
        kl_bias = 0.5 * tf.reduce_sum(log_vp_b - log_vq_b + (tf.pow(mq_b - mp_b, 2) / tf.exp(log_vp_b)) \
                                   + tf.exp(log_vq_b - log_vp_b) - 1)
        
        #log_vp = tf.reshape(log_vp, (-1, 1))
        return kl_normal + kl_bias

    def get_kl_multi(self):
        """
        Compute KL divergence between variational and prior using a multi-layer-network
        """
        kl = 0
        
        # instead of going through the layers, we have to go through the filters/weight matrices
        for key in self.weights.keys():
            kl = kl + self.get_kl(key)
        
        return kl
    
    def get_nelbo(self):
        """ Returns the negative ELBOW, which allows us to minimize instead of maximize. """
        # the kl does not change among samples
        kl = self.get_kl_multi()
        ell, layer_out = self.get_ell()
        # we take the mean instead of the sum to give it the same weight as for the KL-term
        batch_ell = tf.reduce_mean(ell)
        nelbo = kl - batch_ell # * self.N / float(self.M)
        return nelbo, kl, batch_ell, layer_out
    
    def learn(self, learning_rate=0.01, epochs=50):
        """ Our learning procedure """
        optimizer = tf.train.AdamOptimizer(learning_rate)

        ## Set all_variables to contain the complete set of TF variables to optimize
        all_variables = tf.trainable_variables()
        
        ## Define the optimizer
        train_step = optimizer.minimize(self.loss, var_list=all_variables)

        tf.summary.scalar('negative_elbo', self.loss)
        tf.summary.scalar('kl_div', self.kl)
        tf.summary.scalar('ell', self.ell)
        
        merged = tf.summary.merge_all()
        
        train_writer = tf.summary.FileWriter('logs/train', self.session.graph)
        test_writer = tf.summary.FileWriter('logs/test')        
        
        ## Initialize all variables
        init = tf.global_variables_initializer()
        #init_l = tf.local_variables_initializer()

        ## Initialize TF session
        self.session.run(init)
        #self.session.run(init_l)
        
        mean_img = np.mean(mnist.train.images, axis=0)
        
        num_batches = mnist.train.num_examples // self.M

        for i in range(epochs):
            start_time = time.time()
            train_cost = 0
            cum_ell = 0
            cum_kl = 0
            
            old_progress = 0
            for batch_i in range(mnist.train.num_examples // self.M):
                progress = round(float(batch_i) / (mnist.train.num_examples // self.M) * 100)
                if progress % 10 == 0 and progress != old_progress:
                    # print('Progress: ', str(progress) + '%')
                    old_progress = progress

                batch_xs, _ = mnist.train.next_batch(self.M)

                _, loss, ell, kl, summary = self.session.run(
                    [train_step, self.loss, self.ell, self.kl, merged],
                    feed_dict={self.X: batch_xs, self.Y: batch_xs, self.phase: True})
                train_writer.add_summary(summary, i)
                train_cost += loss
                cum_ell += ell
                cum_kl += kl
            

            train_cost /= num_batches
            cum_ell /= num_batches
            cum_kl /= num_batches

            val_cost = self.benchmark(validation=True)
            
            print("   [%.1f] Epoch: %02d | NELBO: %.6f | ELL: %.6f | KL: %.6f | Val. NELBO: %.6f"%(
                time.time()-start_time,i+1, train_cost, cum_ell, cum_kl, val_cost )) 
        
        
        train_writer.close()
        test_writer.close()
        
    def benchmark(self, validation=False):
        if validation:
            benchmark_data = mnist.validation
            label = 'Validation loss:'
        else:
            benchmark_data = mnist.test
            label = 'Test loss:'
        
        cost = 0
        for batch_i in range(benchmark_data.num_examples // self.M):
            batch_xs, _ = benchmark_data.next_batch(self.M)
            cost += self.session.run(self.loss,
                                   feed_dict={self.X: batch_xs, self.Y: batch_xs, self.phase: False})
        return cost / (benchmark_data.num_examples // self.M)
        
    def serialize(self, path):
        saver = tf.train.Saver()
        save_path = saver.save(self.session, path)
        print("Model saved in file: %s" % save_path)
        
    def restore(self, path):
        saver = tf.train.Saver()   
        sess = tf.Session()
        saver.restore(sess, save_path=path)
        self.session = sess
    
    def predict(self, batch):
        outputs = self.layer_out
        return self.session.run(outputs, feed_dict={self.X: batch, self.Y: batch, self.phase: False})
    
    def get_weights(self):
        weights = (self.prior_mean_W, self.log_prior_var_W, self.mean_W, self.log_var_W)
        return self.session.run(weights)
    
    def plot_enc_dec(self, n_examples=10):
        # Plot example reconstructions
        test_xs, _ = mnist.test.next_batch(n_examples)
        recon = self.predict(test_xs)
        fig, axs = plt.subplots(2, n_examples, figsize=(20, 4))
        for example_i in range(n_examples):
            axs[0][example_i].imshow(
                np.reshape(test_xs[example_i, :], (28, 28)))
            axs[1][example_i].imshow(
                np.reshape(
                    np.reshape(recon[example_i, ...], (784,)),
                    (28, 28)))
            plt.gray()
            axs[0][example_i].get_xaxis().set_visible(False)
            axs[0][example_i].get_yaxis().set_visible(False)
            axs[1][example_i].get_xaxis().set_visible(False)
            axs[1][example_i].get_yaxis().set_visible(False)
        plt.show()