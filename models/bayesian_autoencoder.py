import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import matplotlib.pyplot as plt
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

class BayesianAutoencoder(object):
    def __init__(self, neurons_per_layer, batch_size, mc_samples=1, constant_prior=False):
        # SIZES
        tf.reset_default_graph()
        
        self.N = mnist.train.num_examples
        self.layers = len(neurons_per_layer)
        self.neurons_per_layer = neurons_per_layer
        self.M = batch_size
        ## Set the number of Monte Carlo samples as a placeholder so that it can be different for training and test
        self.L =  tf.placeholder(tf.int32)
        # self.L = mc_samples
        
        self.constant_prior = constant_prior
        
        ## Batch data placeholders
        with tf.name_scope('input'):
            self.X = tf.placeholder(tf.float32, shape=[None, neurons_per_layer[0]], name='x-input')
            self.Y = tf.placeholder(tf.float32, shape=[None, neurons_per_layer[-1]], name='y-input')
            
        with tf.name_scope('input_reshape'):
            image_shaped_input = tf.reshape(self.X, [-1, 28, 28, 1])
            tf.summary.image('input', image_shaped_input, 10)
        
        # PRIOR OF WEIGHTS
        self.prior_mean_W, self.log_prior_var_W = self.get_prior_W()
    
        # POSTERIOR OF WEIGHTS
        self.mean_W, self.log_var_W = self.init_posterior_W()
        
        ## Initialize the session
        self.session = tf.Session()
        
        ## Builds whole computational graph with relevant quantities as part of the class
        self.loss, self.kl, self.ell, self.layer_out = self.get_nelbo()

    def get_prior_W(self):
        """
        Define a prior for the weight distribution.
        We assume them to be standard normal iid.
        """
        prior_mean_W = []
        log_prior_var_W = []
        
        for i in range(self.layers - 1):
            d_in = self.neurons_per_layer[i] + 1 # + 1 because of bias weight
            d_out = self.neurons_per_layer[i+1]
            
            with tf.name_scope("layer_" + str(i+1) + "_prior_weights"):
                if self.constant_prior:
                    prior_mean = tf.constant(0.0, shape=[d_in, d_out])
                    log_prior_var = tf.constant(0.0, shape=[d_in, d_out])
                else:
                    prior_mean = tf.Variable(tf.zeros([d_in, d_out]), name="p_W")
                    log_prior_var = tf.Variable(tf.zeros([d_in, d_out]), name="p_W")
                    
                tf.summary.histogram('prior_mean', tf.reshape(prior_mean, [-1]))
                tf.summary.histogram('prior_logvar', tf.reshape(log_prior_var, [-1]))
            
            prior_mean_W.append(prior_mean)
            log_prior_var_W.append(log_prior_var)
        
        return prior_mean_W, log_prior_var_W

    def init_posterior_W(self):
        """
        The (variational) posterior is assumed to be
        drawn from P mutually independent normal distributions.
        Hence, we have a diagonal covariance matrix and only need to store an array.
        """
        mean_W = []
        log_var_W = []
        
        for i in range(self.layers - 1):
            d_in = self.neurons_per_layer[i] + 1 # + 1 because of bias weight
            d_out = self.neurons_per_layer[i+1]

            with tf.name_scope("layer_" + str(i+1) + "_posterior_weights"):
                post_mean = tf.Variable(tf.zeros([d_in, d_out]), name="q_W")
                post_log_var = tf.Variable(tf.zeros([d_in, d_out]), name="q_W")
                tf.summary.histogram('posterior_mean', tf.reshape(post_mean, [-1]))
                tf.summary.histogram('posterior_logvar', tf.reshape(post_log_var, [-1]))
            
            mean_W.append(post_mean)
            log_var_W.append(post_log_var)
            
        return mean_W, log_var_W
    
    def get_std_norm_samples(self, shape):
        """
        Draws N(0,1) samples of dimension [d_in, d_out].
        """
        return tf.random_normal(shape=shape)

    def sample_from_W(self):
        """
        Samples from the variational posterior approximation.
        We draw W-samples for each layer using the reparameterization trick.
        """

        for i in range(self.layers - 1):
            mc_samples = self.L
            d_in = self.neurons_per_layer[i] + 1 # + 1 because of bias weight
            d_out = self.neurons_per_layer[i+1]
            z = self.get_std_norm_samples([mc_samples, d_in, d_out])
            # z = self.get_std_norm_samples([d_in, d_out])
            ## division by 2 to obtain pure standard deviation
            w_from_q = tf.add(tf.multiply(z, tf.exp(self.log_var_W[i] / 2)), self.mean_W[i])

            yield w_from_q
    
    """
    def get_ll(self, Y, output):
        return -tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=output)
        
    """
    def get_ll(self, target, output):
        return tf.reduce_sum(
            target * tf.log(output + 1e-10) + \
            (1 - target) * tf.log(1 - output + 1e-10),
            reduction_indices=[-1]
        )
    
    def get_ell(self, intermediate=0):
        """
        Returns the expected log-likelihood of the lower bound.
        For this we draw L samples from W, compute the log-likelihood for each
        and average the log-likelihoods in the end (expectation approximation).
        This function also includes the entire feedforward pass
        """
        
        # We will generate L output samples
        batch_size = tf.shape(self.X)[0]
        d_in = tf.shape(self.X)[-1]
        outputs = tf.multiply(tf.ones([self.L, batch_size, d_in]), self.X)
        
        # cum_out = 0
        # cum_ll = 0
        
        # for i in range(self.L):        
            # outputs = self.X

        # Go through each layer (one weight matrix at a time)
        # and compute the (intermediate) output
        j = 0

        for weight_matrix in self.sample_from_W():
            outputs = tf.matmul(outputs, weight_matrix[:,1:,:])
            bias = tf.expand_dims(weight_matrix[:,0,:], 1)
            outputs = outputs + bias
            # outputs = tf.matmul(outputs, weight_matrix[1:,:]) + weight_matrix[0,:]
            tf.summary.histogram('activations', outputs)

            # if last layer is reached, do not use transfer function (softmax later on)
            if j == (self.layers - 2):
                outputs = tf.sigmoid(outputs)
            else:
                outputs = tf.nn.tanh(outputs)

            tf.summary.histogram('outputs', outputs)

            j += 1

            if j == intermediate:
                break

        ll = self.get_ll(self.Y, outputs)
        ell = tf.reduce_mean(ll, 0)
        avg_out = tf.reduce_mean(outputs, 0)
        # cum_ll += ll
        # cum_out += outputs

        return ell, avg_out
        # return cum_ll / self.L, cum_out / self.L

    def get_kl(self, mean_W, log_var_W, prior_mean_W, log_prior_var_W):
        """
        KL[q || p] returns the KL-divergence between the prior p and the variational posterior q.
        :param mq: vector of means for q
        :param log_vq: vector of log-variances for q
        :param mp: vector of means for p
        :param log_vp: vector of log-variances for p
        :return: KL divergence between q and p
        """
        mq = mean_W
        log_vq = log_var_W
        mp = prior_mean_W
        log_vp = log_prior_var_W
        
        return 0.5 * tf.reduce_sum(log_vp - log_vq + (tf.square(mq - mp) / tf.exp(log_vp)) + tf.exp(log_vq - log_vp) - 1)

    def get_kl_multi(self):
        """
        Compute KL divergence between variational and prior using a multi-layer-network
        """
        kl = 0
        
        for i in range(self.layers - 1):
            kl = kl + self.get_kl(
                        self.mean_W[i],
                        self.log_var_W[i],
                        self.prior_mean_W[i],
                        self.log_prior_var_W[i]
            )
        
        return kl
    
    def get_nelbo(self):
        """ Returns the negative ELBOW, which allows us to minimize instead of maximize. """
        # the kl does not change among samples
        kl = self.get_kl_multi()
        ell, layer_out = self.get_ell()
        # we take the mean instead of the sum to give it the same weight as for the KL-term
        batch_ell = tf.reduce_mean(ell)
        nelbo = kl - tf.reduce_sum(ell) * self.N / float(self.M)
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

        ## Initialize TF session
        self.session.run(init)
        
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
                    feed_dict={self.X: batch_xs, self.Y: batch_xs, self.L: 1})
                train_writer.add_summary(summary, i)
                train_cost += loss
                cum_ell += ell
                cum_kl += kl
            
            train_cost /= num_batches
            cum_ell /= num_batches
            cum_kl /= num_batches

            val_cost = self.benchmark(validation=True)
            
            print("   [%.1f] Epoch: %02d | NELBO: %.6f | ELL: %.6f | KL: %.6f | Val. ELL: %.6f"%(
                time.time()-start_time,i+1, train_cost, cum_ell, cum_kl, val_cost ))        
        
        train_writer.close()
        test_writer.close()
    
    def benchmark(self, validation=False):
        # TEST LOG LIKELIHOOD
        if validation:
            benchmark_data = mnist.validation
        else:
            benchmark_data = mnist.test
        
        total_batch = benchmark_data.num_examples // self.M
        ell = 0
        for batch_i in range(total_batch):
            batch_xs, _ = benchmark_data.next_batch(self.M)
            c = self.session.run(self.ell,
                   feed_dict={self.X: batch_xs, self.Y: batch_xs, self.L: 10})
            ell+= c/total_batch
        
        return ell
        
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
        return self.session.run(outputs, feed_dict={self.X: batch, self.Y: batch, self.L: 10})
    
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
