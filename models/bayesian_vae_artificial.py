import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import matplotlib.pyplot as plt
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

class BayesianVAEArtificial(object):
    def __init__(self,
                 n_inputs=784,
                 n_neurons_encoder = [2048, 256],
                 n_latent=2,
                 n_neurons_decoder = [256, 2048],
                 batch_size = 128,
                 activation = tf.nn.tanh,
                 mc_samples = 1,
                 constant_prior=False):
        
        # SIZES
        tf.reset_default_graph()
        
        self.N = mnist.train.num_examples
        self.n_inputs = n_inputs
        self.n_encoder = n_neurons_encoder
        self.n_decoder = n_neurons_decoder
        self.n_latent = n_latent
        self.length_encoder = len(n_neurons_encoder)
        self.length_decoder = len(n_neurons_decoder)
        self.layers = self.length_encoder + self.length_decoder + 1 
        self.activ = activation
        
        self.M = batch_size
        ## Set the number of Monte Carlo samples as a placeholder so that it can be different for training and test
        # self.L =  tf.placeholder(tf.int32)
        self.L = mc_samples
        
        self.constant_prior = constant_prior
        
        ## Batch data placeholders
        with tf.name_scope('input'):
            self.X = tf.placeholder(tf.float32, shape=[None, self.n_inputs], name='x-input')
            self.Y = tf.placeholder(tf.float32, shape=[None, self.n_inputs], name='y-input')
            
        with tf.name_scope('input_reshape'):
            image_shaped_input = tf.reshape(self.X, [-1, 28, 28, 1])
            tf.summary.image('input', image_shaped_input, 10)
        
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
        
            # POSTERIORS
            self.weights[name]['post_mean'] = tf.Variable(tf.zeros(shape), name="post_mean")
            self.weights[name]['post_mean_bias'] = tf.Variable(tf.zeros(bias_shape), name="post_mean_bias")
            self.weights[name]['post_log_var'] = tf.Variable(tf.zeros(shape), name="post_log_var")
            self.weights[name]['post_log_var_bias'] = tf.Variable(tf.zeros(bias_shape), name="post_log_var_bias")
            
    def initialize_model_weights(self):
        with tf.name_scope("Encoder_layer_weights"):
            
            self.add_weights([self.n_inputs, self.n_encoder[0]], [self.n_encoder[0]], 'W_enc_0')
            for i in range(1, self.length_encoder):
                self.add_weights([self.n_encoder[i-1], self.n_encoder[i]], [self.n_encoder[i]], 'W_enc_' + str(i))
        
        with tf.name_scope("Latent_layer_weights"):
            self.add_weights([self.n_encoder[-1], self.n_latent], [self.n_latent], 'W_latent_mu')
            self.add_weights([self.n_encoder[-1], self.n_latent], [self.n_latent], 'W_latent_log_sigma')     
            
        with tf.name_scope("Decoder_layer_weights"):
            self.add_weights([self.n_latent, self.n_decoder[0]], [self.n_decoder[0]], 'W_dec_0')
            for i in range(1, self.length_decoder):
                self.add_weights([self.n_decoder[i-1], self.n_decoder[i]], [self.n_decoder[i]], 'W_dec_' + str(i))
            self.add_weights([self.n_decoder[-1], self.n_inputs], [self.n_inputs], 'W_dec_' + str(self.length_decoder))
    
    def get_std_norm_samples(self, shape):
        """
        Draws N(0,1) samples of dimension [d_in, d_out].
        """
        return tf.random_normal(shape=shape)

    def sample_from_W(self, layer_name, bias=False):
        """
        Samples from the variational posterior approximation.
        We draw W-samples for each layer using the reparameterization trick.
        """

        if bias:
            mean_W = self.weights[layer_name]['post_mean_bias']
            log_var_W = self.weights[layer_name]['post_log_var_bias']
        else:
            mean_W = self.weights[layer_name]['post_mean']
            log_var_W = self.weights[layer_name]['post_log_var']
        
        # mc_samples = self.L
        shape = mean_W.get_shape().as_list()
        # shape.insert(0, mc_samples)
        
        z = self.get_std_norm_samples(shape)
        ## division by 2 to obtain pure standard deviation
        w_from_q = tf.add(tf.multiply(z, tf.exp(log_var_W / 2)), mean_W)
        
        return w_from_q
    
    def sample_from_Z(self, z_mu, z_log_sigma):
        """
        Samples from the posterior of the variational latent space.
        We draw samples using the reparameterization trick.
        """
        return z_mu + tf.exp(z_log_sigma) * self.get_std_norm_samples([tf.shape(self.X)[0], self.n_latent])
    
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
        # batch_size = tf.shape(self.X)[0]
        # d_in = tf.shape(self.X)[-1]
        # outputs = tf.multiply(tf.ones([self.L, batch_size, d_in]), self.X)
        
        cum_out = 0
        cum_ll = 0
        
        encodigns = []
        for i in range(self.L):

            # ENCODER
            for i in range(self.length_encoder):
                W = self.sample_from_W('W_enc_' + str(i))
                W_bias = self.sample_from_W('W_enc_' + str(i), bias=True)
                W_bias = tf.expand_dims(W_bias, 1)            
                outputs = self.activ(tf.matmul(outputs, W) + W_bias)

            encodings.append(outputs)
            ##### SOLVE tensor mc samples first!!! Then it becomes easier!
            # LATENT
            W = self.sample_from_W('W_latent_mu')
            W_bias = self.sample_from_W('W_latent_mu', bias=True)
            W_bias = tf.expand_dims(W_bias, 1)
            z_mu = tf.matmul(outputs, W) + W_bias
            W = self.sample_from_W('W_latent_log_sigma')
            W_bias = self.sample_from_W('W_latent_log_sigma', bias=True)
            W_bias = tf.expand_dims(W_bias, 1)
            z_log_sigma = 0.5 * (tf.matmul(outputs, W) + W_bias)

            # Sample from posterior
            z = self.sample_from_Z(z_mu, z_log_sigma)

            # DECODER
            W = self.sample_from_W('W_dec_' + str(0))
            W_bias = self.sample_from_W('W_dec_' + str(0), bias=True)
            W_bias = tf.expand_dims(W_bias, 1)
            outputs = self.activ(tf.matmul(z, W) + W_bias)
            for i in range(1, self.length_decoder):
                W = self.sample_from_W('W_dec_' + str(i))
                W_bias = self.sample_from_W('W_dec_' + str(i), bias=True)
                W_bias = tf.expand_dims(W_bias, 1)
                outputs = self.activ(tf.matmul(outputs, W) + W_bias)

            W = self.sample_from_W('W_dec_' + str(self.length_decoder))
            W_bias = self.sample_from_W('W_dec_' + str(self.length_decoder), bias=True)
            W_bias = tf.expand_dims(W_bias, 1)
            outputs = tf.nn.sigmoid(tf.matmul(outputs, W) + W_bias)

            ll = self.get_ll(self.Y, outputs)
            cum_ll += ll
            cum_out += outputs
            # avg_out = tf.reduce_mean(outputs, 0)

            # return tf.reduce_mean(ll, 0), avg_out
        return cum_ll / self.L, cum_out / self.L

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
                    feed_dict={self.X: batch_xs, self.Y: batch_xs})
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
                                   feed_dict={self.X: batch_xs, self.Y: batch_xs})
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
        return self.session.run(outputs, feed_dict={self.X: batch, self.Y: batch})
    
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
